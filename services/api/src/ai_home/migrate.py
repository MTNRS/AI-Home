"""Run packaged schema migrations using AI_HOME_DATABASE_URL and Settings."""
import argparse

from alembic import command
from alembic.config import Config


def migration_config():
    config = Config()
    config.set_main_option("script_location", "ai_home:migrations")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upgrade", "current", "check"))
    args = parser.parse_args()
    config = migration_config()
    if args.action == "upgrade":
        command.upgrade(config, "head")
    elif args.action == "check":
        command.check(config)
    else:
        command.current(config)


if __name__ == "__main__":
    main()
