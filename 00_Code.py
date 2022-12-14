# import allan
# import kevin
# import alexandre
# import aymane
import sqlite3
from sqlite3 import OperationalError


# schema_str = open("05_movie.sql","r").read()
connection = sqlite3.connect("data.db")
cur = connection.cursor()

cur.execute("SELECT title FROM movies WHERE vote_average==5;")
print(cur.fetchall())

connection.close()