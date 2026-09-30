.PHONY: up down prepare train promote api ui test

up:
	docker compose up -d

down:
	docker compose down -v

prepare:
	docker compose run --rm trainer python /data/prepare.py

train:
	docker compose run --rm trainer python train.py

promote:
	docker compose run --rm trainer python promote.py

api:
	docker compose up -d --build api

ui:
	docker compose up -d --build ui

test:
	docker compose run --rm trainer pytest -q
