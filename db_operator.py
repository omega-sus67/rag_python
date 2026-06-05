from sqlalchemy import create_engine, Column, Integer, String, Boolean, text, select, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from fastapi import FastAPI, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker,create_async_engine
from pdf_operator import Document
from pydantic import BaseModel
import hashlib
from pgvector.sqlalchemy import Vector

ASYNC_DATABASE_URL = "postgresql+asyncpg://postgres:password@localhost:5433/tutorial_db"

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=True,
)

AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=async_engine)

Base = declarative_base()

class dbFile(Base):
    __tablename__ = "files"
    
    id = Column(String , primary_key=True , index=True)
    title = Column(String , nullable=False)
    extracted_text = Column(String , nullable=False)

class dbChunk(Base):
    __tablename__ = "doc_chunks"

    id = Column(String, primary_key=True, index=True)
    file_id = Column(String , ForeignKey("files.id"))
    chunk_index = Column(Integer , nullable=False)
    text_data = Column(String , nullable=False)
    embeddings = Column(Vector(384))


async def saveDocument(doc : Document):
    uniqueID = hashlib.sha256(doc.extracted_text.encode('utf-8')).hexdigest()

    async with AsyncSessionLocal() as session:
        #checks for duplication and also assigns a unique id to the document based on its hash value(calculate on the basis of the text_extracted from the pdf)
        chk = select(dbFile).where(dbFile.id == uniqueID)
        res = await session.execute(chk)

        if res.scalar():
            raise HTTPException(status_code=400 , detail="File already exists")

        db_doc = dbFile(
            id = uniqueID,
            title = doc.title,
            extracted_text = doc.extracted_text
        )
        session.add(db_doc)
        await session.commit()
        await session.refresh(db_doc)
        return db_doc

async def fetchDoc(id : str):
    async with AsyncSessionLocal() as session:
        chk = select(dbFile).where(dbFile.id == id)
        res = await session.execute(chk)
        doc = res.scalar()
        if not doc:
            raise HTTPException(status_code=404 , detail="File not found")
        return doc

async def createTable():
    temp_url = "postgresql+asyncpg://postgres:password@localhost:5433/postgres"
    async_temp_engine = create_async_engine(temp_url, isolation_level="AUTOCOMMIT")
    
    async with async_temp_engine.connect() as conn:
        res = await conn.execute(text("SELECT 1 FROM pg_database WHERE datname='tutorial_db'"))
        if not res.scalar():
            print("Database 'tutorial_db' does not exist. Creating it...")
            await conn.execute(text("CREATE DATABASE tutorial_db"))
        else:
            print("Database 'tutorial_db' already exists.")
            
    await async_temp_engine.dispose()

    # 2. Now connect to 'tutorial_db' and create tables
    async with async_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)
