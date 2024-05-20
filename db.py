import mysql.connector
from mysql.connector import Error

hostname = "05j.h.filess.io"
database = "toddlerASD_aidwhatmix"
port = "3307"
username = "toddlerASD_aidwhatmix"
password = "25b540cda0c6795eda70e2127651638af1c1174c"
sql_file = "toddler_asd_dataset.sql"

try:
    connection = mysql.connector.connect(host=hostname, database=database, user=username, password=password, port=port)
    if connection.is_connected():
        cursor = connection.cursor()
        with open(sql_file, 'r') as file:
            sql_commands = file.read().split(';')
            for command in sql_commands:
                cursor.execute(command)
        print("SQL file imported successfully.")

except Error as e:
    print("Error while connecting to MySQL", e)

finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")
