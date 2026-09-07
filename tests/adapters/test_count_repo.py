from unittest.mock import MagicMock, patch
from counter.adapters.count_repo import (
    CountInMemoryRepo,
    CountMongoDBRepo,
    CountPostgresRepo,
)
from counter.domain.models import ObjectCount


class TestCountInMemoryRepo:
    def test_empty_repo_reads_empty(self):
        repo = CountInMemoryRepo()
        assert repo.read_values() == []
        assert repo.read_values(["cat"]) == []

    def test_update_and_read_values(self):
        repo = CountInMemoryRepo()
        repo.update_values([ObjectCount("cat", 2), ObjectCount("dog", 1)])

        values = repo.read_values()
        assert len(values) == 2
        value_map = {v.object_class: v.count for v in values}
        assert value_map["cat"] == 2
        assert value_map["dog"] == 1

    def test_update_accumulates_counts(self):
        repo = CountInMemoryRepo()
        repo.update_values([ObjectCount("cat", 2)])
        repo.update_values([ObjectCount("cat", 3)])

        values = repo.read_values(["cat"])
        assert len(values) == 1
        assert values[0].count == 5

    def test_read_filtered_classes(self):
        repo = CountInMemoryRepo()
        repo.update_values([ObjectCount("cat", 2), ObjectCount("dog", 3), ObjectCount("bird", 1)])

        values = repo.read_values(["cat", "bird", "unknown"])
        assert len(values) == 2
        value_map = {v.object_class: v.count for v in values}
        assert value_map["cat"] == 2
        assert value_map["bird"] == 1


class TestCountPostgresRepo:
    def test_postgres_repo_uri_normalization(self):
        with patch("counter.adapters.count_repo.create_engine") as mock_create_engine:
            _ = CountPostgresRepo(uri="postgresql://user:pass@localhost:5432/db")
            mock_create_engine.assert_called_with(
                "postgresql+psycopg://user:pass@localhost:5432/db", pool_pre_ping=True
            )


class TestCountMongoDBRepo:
    @patch("counter.adapters.count_repo.MongoClient")
    def test_mongo_repo_read_and_update(self, mock_mongo_client_cls):
        mock_client = MagicMock()
        mock_mongo_client_cls.return_value = mock_client
        mock_db = MagicMock()
        mock_col = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_col
        mock_col.find.return_value = [
            {"object_class": "cat", "count": 3},
            {"object_class": "dog", "count": 1},
        ]

        repo = CountMongoDBRepo(host="localhost", port=27017, database="test_db")
        results = repo.read_values()

        assert len(results) == 2
        assert results[0].object_class == "cat"
        assert results[0].count == 3

        repo.update_values([ObjectCount("cat", 2)])
        mock_col.update_one.assert_called_once_with(
            {"object_class": "cat"},
            {"$inc": {"count": 2}},
            upsert=True,
        )
