.PHONY: up
up:
	    docker compose up -d --build

.PHONY: exec
exec:
	    docker compose exec b4_research bash