import sqlite3  
  
con = sqlite3.connect("docs.db")  
print("Database opened successfully")  
  
con.execute("create table tab(id INTEGER PRIMARY KEY AUTOINCREMENT, doc_text TEXT NOT NULL)")  
  
print("Table created successfully")  
  
con.close()  