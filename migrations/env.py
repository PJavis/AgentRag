# migrations/env.py
# migrations/env.py
import sys
import os
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path
project_root = Path(__file__).resolve().parents[1]  # lên 1 cấp từ migrations/
sys.path.insert(0, str(project_root))

import asyncio
from logging.config import fileConfig

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import pool

from alembic import context

# Alembic Config object
config = context.config

# Setup logging từ alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ==================== PHẦN QUAN TRỌNG ====================
# Import settings và Base (sẽ tạo Base ở bước sau)
from src.agentrag.config import settings
from src.agentrag.database import Base
from src.agentrag.database.models import *

# Tạm thời để target_metadata = None vì chưa có model
# Sau khi tạo models sẽ thay bằng Base.metadata
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    url = settings.DATABASE_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations_online() -> None:
    """Run migrations in 'online' mode - dùng sync engine cho Alembic CLI"""
    from sqlalchemy import create_engine  # import sync engine

    connectable = create_engine(
        settings.DATABASE_URL,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    do_run_migrations_online()