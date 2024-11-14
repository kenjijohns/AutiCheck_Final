import mysql.connector
from mysql.connector import Error

hostname = "z1tqq.h.filess.io"
database= "ToddlerASD_strangeas"
port = "3307"
username = "ToddlerASD_strangeas"
password = "5919ad83daeb3c9694baae64f212a4ede24abfe7"
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
