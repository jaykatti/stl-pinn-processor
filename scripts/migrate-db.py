#!/usr/bin/env python3
"""
Fixed database migration script for STL-PINN Processor
"""
import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

async def run_migrations():
    """Run database migrations with better error handling"""
    try:
        # Import settings with error handling
        try:
            from config.settings import settings
            print("✅ Settings loaded successfully")
        except Exception as e:
            print(f"❌ Settings loading failed: {e}")
            print("Creating minimal configuration...")
            
            # Create minimal settings for migration
            class MinimalSettings:
                database_url = os.getenv("DATABASE_URL", "sqlite:///./stl_pinn.db")
            
            settings = MinimalSettings()
            print(f"Using database: {settings.database_url}")
        
        print("🗄️ Running database migrations...")
        print(f"Database URL: {settings.database_url}")
        
        if settings.database_url.startswith("postgresql"):
            await migrate_postgresql(settings.database_url)
        elif settings.database_url.startswith("sqlite"):
            await migrate_sqlite(settings.database_url)
        else:
            print("⚠️  Unknown database type, skipping migrations")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

async def migrate_postgresql(database_url):
    """PostgreSQL migrations"""
    try:
        import asyncpg
        from urllib.parse import urlparse
        
        parsed = urlparse(database_url)
        
        try:
            conn = await asyncpg.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path[1:] if parsed.path else 'postgres'
            )
            
            # Create basic tables
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(255) UNIQUE NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    input_file_path TEXT,
                    output_file_path TEXT,
                    parameters JSONB,
                    metadata JSONB
                );
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status);
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON processing_jobs(created_at);
            ''')
            
            await conn.close()
            print("✅ PostgreSQL migrations completed")
            
        except Exception as e:
            if "does not exist" in str(e).lower():
                print("⚠️  Database does not exist. Please create it first:")
                print(f"  createdb {parsed.path[1:] if parsed.path else 'stl_pinn'}")
            else:
                print(f"❌ PostgreSQL migration failed: {e}")
            
    except ImportError:
        print("⚠️  asyncpg not installed. Install with: pip install asyncpg")
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")

async def migrate_sqlite(database_url):
    """SQLite migrations"""
    try:
        import aiosqlite
        
        # Extract database path from URL
        db_path = database_url.replace("sqlite:///", "")
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        async with aiosqlite.connect(db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    input_file_path TEXT,
                    output_file_path TEXT,
                    parameters TEXT,
                    metadata TEXT
                );
            ''')
            
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status);
            ''')
            
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON processing_jobs(created_at);
            ''')
            
            await db.commit()
            print("✅ SQLite migrations completed")
            
    except ImportError:
        print("⚠️  aiosqlite not installed. Install with: pip install aiosqlite")
    except Exception as e:
        print(f"❌ SQLite migration failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_migrations())