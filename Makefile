.PHONY: setup up down check logs

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	cp -n .env.example .env || true

up:
	docker compose up -d

down:
	docker compose down

check:
	python scripts/check_infra.py

logs:
	docker compose logs -f --tail=50

.PHONY: setup up down check logs init-db land test

init-db:
	python scripts/init_db.py

land:
	python scripts/run_landing.py

test:
	python -m pytest -q
