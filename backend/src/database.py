import os
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase,sessionmaker
url=os.getenv('DATABASE_URL','mysql+pymysql://quiet:quiet-demo@db:3306/quiet')
options={'pool_pre_ping':True}
# MySQL显式READ COMMITTED；SQLite（仅本地测试）不支持该隔离级别参数。
if not url.startswith('sqlite'):options['isolation_level']='READ COMMITTED'
# SQLite仅用于本地/自动测试，允许跨线程连接；生产使用MySQL。
else:options['connect_args']={'check_same_thread':False}
engine=create_engine(url,**options)
class Base(DeclarativeBase): pass
Session=sessionmaker(engine,expire_on_commit=False)
def session():
    with Session() as db:
        try: yield db
        except Exception:
            db.rollback()
            raise
