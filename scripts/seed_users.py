"""Create the API users listed under `users:` in config/company.yaml.

Passwords are generated randomly and shown ONCE (only a salted hash is stored). Existing users
are left alone unless you pass --reset.

Run:  python scripts/seed_users.py                     # new users get random passwords
      python scripts/seed_users.py --password demo1234 # same password for everyone (local demos only)
      python scripts/seed_users.py --reset --only alice
      python scripts/seed_users.py --disable bob       # lock an account (tokens stop working at once)
"""
import argparse
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import init_db
from core.settings import company
from serving.security import hash_password
from serving.users import SqlUserStore

parser = argparse.ArgumentParser()
parser.add_argument("--password", help="use this password for every user created/reset")
parser.add_argument("--reset", action="store_true", help="re-set password of users that already exist")
parser.add_argument("--only", help="only this username")
parser.add_argument("--disable", metavar="USER", help="disable an account and exit")
parser.add_argument("--enable", metavar="USER", help="re-enable an account and exit")
args = parser.parse_args()

init_db()
store = SqlUserStore()

for flag, value in (("--disable", True), ("--enable", False)):
    name = getattr(args, flag.strip("-"))
    if name:
        sys.exit(f"{flag[2:].capitalize()}d {name}" if store.set_disabled(name, value) else f"No such user: {name}")

roles = company["roles"]
for entry in company["users"]:
    name, role = entry["username"], entry["role"]
    if args.only and name != args.only.lower():
        continue
    if role not in roles:
        print(f"  skipped {name}: role '{role}' is not defined under roles:")
        continue
    if store.get(name) and not args.reset:
        print(f"  exists  {name} ({role})  - unchanged")
        continue
    password = args.password or secrets.token_urlsafe(9)
    store.upsert(name, hash_password(password), role)
    print(f"  created {name} ({role})  password: {password}")
print("\nPasswords are shown only now. Store them somewhere safe.")
