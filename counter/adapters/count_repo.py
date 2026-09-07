import logging
from typing import Any, List, Optional

from pymongo import MongoClient
from sqlalchemy import BigInteger, Column, Engine, MetaData, String, Table, create_engine
from sqlalchemy.dialects.postgresql import insert as pg_insert

from counter.domain.models import ObjectCount
from counter.domain.ports import ObjectCountRepo

logger = logging.getLogger(__name__)


class CountInMemoryRepo(ObjectCountRepo):
    """Thread-safe in-memory repository implementation of ObjectCountRepo."""

    def __init__(self) -> None:
        self.store: dict[str, ObjectCount] = dict()

    def read_values(self, object_classes: Optional[List[str]] = None) -> List[ObjectCount]:
        if object_classes is None:
            return list(self.store.values())
        return [self.store[cls_name] for cls_name in object_classes if cls_name in self.store]

    def update_values(self, new_values: List[ObjectCount]) -> None:
        for new_object_count in new_values:
            key = new_object_count.object_class
            if key in self.store:
                self.store[key] = ObjectCount(key, self.store[key].count + new_object_count.count)
            else:
                self.store[key] = ObjectCount(key, new_object_count.count)


class CountMongoDBRepo(ObjectCountRepo):
    """MongoDB repository implementation of ObjectCountRepo with connection reuse."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        database: str = "prod_counter",
        uri: Optional[str] = None,
    ) -> None:
        self._client: MongoClient[Any]
        if uri:
            self._client = MongoClient(uri)
        else:
            self._client = MongoClient(host, port)
        self._database_name = database
        self._db = self._client[self._database_name]
        self._collection = self._db["counter"]

    def read_values(self, object_classes: Optional[List[str]] = None) -> List[ObjectCount]:
        query = {"object_class": {"$in": object_classes}} if object_classes else {}
        counters = self._collection.find(query)
        return [ObjectCount(c["object_class"], c["count"]) for c in counters]

    def update_values(self, new_values: List[ObjectCount]) -> None:
        for value in new_values:
            self._collection.update_one(
                {"object_class": value.object_class},
                {"$inc": {"count": value.count}},
                upsert=True,
            )


class CountPostgresRepo(ObjectCountRepo):
    """PostgreSQL/Relational database repository implementation of ObjectCountRepo using SQLAlchemy."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        uri: Optional[str] = None,
        engine: Optional[Engine] = None,
    ) -> None:
        if engine is not None:
            self.engine = engine
        elif uri:
            # Normalize postgres:// to postgresql+psycopg:// if needed
            if uri.startswith("postgres://"):
                uri = uri.replace("postgres://", "postgresql+psycopg://", 1)
            elif (
                uri.startswith("postgresql://") and "+psycopg" not in uri and "+asyncpg" not in uri
            ):
                uri = uri.replace("postgresql://", "postgresql+psycopg://", 1)
            self.engine = create_engine(uri, pool_pre_ping=True)
        else:
            h = host or "localhost"
            p = port or 5432
            db = database or "counter_db"
            u = user or "postgres"
            pwd = password or "postgres"
            connection_url = f"postgresql+psycopg://{u}:{pwd}@{h}:{p}/{db}"
            self.engine = create_engine(connection_url, pool_pre_ping=True)

        self._metadata = MetaData()
        self._table = Table(
            "object_counts",
            self._metadata,
            Column("object_class", String(255), primary_key=True),
            Column("count", BigInteger, nullable=False, default=0),
        )
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they do not exist."""
        try:
            self._metadata.create_all(self.engine)
        except Exception as e:
            logger.warning(f"Could not auto-create schema: {e}")

    def read_values(self, object_classes: Optional[List[str]] = None) -> List[ObjectCount]:
        with self.engine.connect() as conn:
            query = self._table.select()
            if object_classes is not None:
                query = query.where(self._table.c.object_class.in_(object_classes))
            rows = conn.execute(query).fetchall()
            return [
                ObjectCount(
                    object_class=str(row._mapping["object_class"]),
                    count=int(row._mapping["count"]),
                )
                for row in rows
            ]

    def update_values(self, new_values: List[ObjectCount]) -> None:
        if not new_values:
            return

        with self.engine.begin() as conn:
            for val in new_values:
                dialect = self.engine.dialect.name
                if dialect == "postgresql":
                    pg_stmt = (
                        pg_insert(self._table)
                        .values(object_class=val.object_class, count=val.count)
                        .on_conflict_do_update(
                            index_elements=["object_class"],
                            set_={"count": self._table.c.count + val.count},
                        )
                    )
                    conn.execute(pg_stmt)
                else:
                    raise RuntimeError(
                        f"CountPostgresRepo requires a PostgreSQL engine, got {dialect!r}"
                    )
