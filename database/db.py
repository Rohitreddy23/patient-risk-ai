import sqlite3
import os

DB_PATH = os.path.join("database", "database.db")

def connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def create_tables():
    conn = connect()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        username TEXT PRIMARY KEY,
        password TEXT,
        role TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        age INTEGER,
        bp INTEGER,
        glucose INTEGER,
        cholesterol INTEGER,
        risk TEXT,
        score REAL
    )
    """)

    conn.commit()
    conn.close()


def add_user(u, p, r):
    conn = connect()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?,?,?)", (u.strip(), p.strip(), r))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()


def login_user(u, p):
    conn = connect()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (u.strip(), p.strip()))
    user = c.fetchone()
    conn.close()
    return user


def save_history(u, age, bp, gl, ch, risk, score):
    conn = connect()
    c = conn.cursor()
    c.execute("""
    INSERT INTO history(username, age, bp, glucose, cholesterol, risk, score)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (u, age, bp, gl, ch, risk, score))
    conn.commit()
    conn.close()


def get_history(user, role):
    conn = connect()
    c = conn.cursor()

    if role == "Admin":
        c.execute("SELECT * FROM history")
    else:
        c.execute("SELECT * FROM history WHERE username=?", (user,))

    data = c.fetchall()
    conn.close()
    return data