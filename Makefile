.PHONY: db-push db-sync db-status

db-push:
	uv run forecast-db push

db-sync:
	uv run forecast-db sync

db-status:
	uv run forecast-db status
