# import allan
# import kevin
# import alexandre
# import aymane
import sqlite3
from sqlite3 import OperationalError


# schema_str = open("05_movie.sql","r").read()
connection = sqlite3.connect("movies.db")
cur = connection.cursor()


def create(cur, sqlpath):
    with open(sqlpath, encoding="utf-8") as fp:
        cur.executescript(fp.read())  # or con.executescript


create(cur, "01_reference_data.sql")
create(cur, "02_keyword.sql")
create(cur, "03_person.sql")
create(cur, "04_production_company")
create(cur, "05_movie.sql")
create(cur, "06_movie_cast.sql")
create(cur, "07_movie_company.sql")
create(cur, "08_movie_crew.sql")
create(cur, "09_movie_genres.sql")
create(cur, "10_movie_keywords.sql")
create(cur, "11_movie_languages.sql")
create(cur, "12_production_country.sql")

connection.commit()


"""
conn = sqlite3.connect('data.db')
c = conn.cursor()

#execute("05_movie.sql")
# For each of the 3 tables, query the database and print the contents
for table in ['movies']:
    # Plug in the name of the table into SELECT * query
    result = c.execute("SELECT * FROM {};".format(table))

    # Get all rows.
    rows = result.fetchall()

    # \n represents an end-of-line
    print("\n--- TABLE ", table, "\n")

    # This will print the name of the columns, padding each name up
    # to 22 characters. Note that comma at the end prevents new lines
    for desc in result.description:
        print(desc[0].rjust(22, ' '))

    # End the line with column names
    print("")
    for row in rows:
        for value in row:
            # Print each value, padding it up with ' ' to 22 characters on the right
            print(str(value).rjust(22, ' ')),
        # End the values from the row
        print("")

c.close()
conn.close()
"""
