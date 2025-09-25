#!/usr/bin/env python3
"""
Database migration script for STL-PINN Processor
"""
import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

async def run_migrations():
    """Run database migrations"""
    try:
        # Import after adding src to path
        from config.settings import settings
        
        print("🗄️ Running database migrations...")
        print(f"Database URL: {settings.database_url}")
        
        # For now, just ensure database connection works
        # In a full implementation, you would use Alembic or similar
        
        if settings.database_url.startswith("postgresql"):
            import asyncpg
            
            # Parse database URL
            from urllib.parse import urlparse
            parsed = urlparse(settings.database_url)
            
            try:
                conn = await asyncpg.connect(
                    host=parsed.hostname,
                    port=parsed.port,
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path[1:] if parsed.path else 'postgres'
                )
                
                # Create basic tables (example)
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
                print(f"❌ Database migration failed: {e}")
                sys.exit(1)
        
        else:
            print("ℹ️ Using SQLite - no migrations needed")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure to install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_migrations())