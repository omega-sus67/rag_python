import asyncio
from app.db.database_manager import Base, DatabaseManager

async def reset():
    db = DatabaseManager()
    async with db.engine.begin() as conn:
        print("Wiping all tables...")
        await conn.run_sync(Base.metadata.drop_all)
        print("Tables wiped! Rebuilding database structure...")
        await conn.run_sync(Base.metadata.create_all)
        print("Database reset successfully.")

if __name__ == "__main__":
    asyncio.run(reset())
