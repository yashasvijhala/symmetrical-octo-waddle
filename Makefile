.PHONY: db-push db-sync db-status seed-demo worker-cpu worker-gpu

db-push:
	uv run forecast-db push

db-sync:
	uv run forecast-db sync

db-status:
	uv run forecast-db status

seed-demo:
	uv run forecast-seed-demo

worker-cpu:
	uv run forecast-worker --kind cpu

worker-gpu:
	uv run forecast-worker --kind gpu
