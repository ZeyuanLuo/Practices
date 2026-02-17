import sqlite3
import os


def initialize_database():

    # 删除已存在的数据库文件
    if os.path.exists('Shop_Exercise.db'):
        os.remove('Shop_Exercise.db')
        print("Old document cleaned")

    conn = sqlite3.connect('Shop_Exercise.db')
    cursor = conn.cursor()

    try:
        # create client table
        cursor.execute('''
            CREATE TABLE client (
                ID_client INTEGER PRIMARY KEY,
                surname TEXT NOT NULL,
                name TEXT NOT NULL,
                city TEXT,
                discount REAL CHECK (discount >= 0 AND discount <= 100)
            )
        ''')

        # create agent table
        cursor.execute('''
            CREATE TABLE agent (
                ID_agent INTEGER PRIMARY KEY,
                surname TEXT NOT NULL,
                name TEXT NOT NULL,
                zone TEXT,
                supervisor INTEGER,
                commission REAL CHECK (commission >= 0 AND commission <= 100),
                FOREIGN KEY (supervisor) REFERENCES agent(ID_agent) ON DELETE SET NULL
            )
        ''')

        # create order table
        cursor.execute('''
            CREATE TABLE order_table (
                num_order INTEGER PRIMARY KEY,
                id_client INTEGER NOT NULL,
                id_agent INTEGER NOT NULL,
                date_order TEXT NOT NULL,
                product TEXT NOT NULL,
                amount REAL CHECK (amount > 0),
                FOREIGN KEY (id_client) REFERENCES client(ID_client) ON DELETE CASCADE,
                FOREIGN KEY (id_agent) REFERENCES agent(ID_agent) ON DELETE RESTRICT
            )
        ''')

        conn.commit()
        print("database table created successfully!！")

    except sqlite3.Error as e:
        print(f"error on database table creation: {e}")
    finally:
        conn.close()


def insert_sample_data():
    """Insert examples"""
    conn = sqlite3.connect('Shop_Exercise.db')
    cursor = conn.cursor()

    try:
        clients = [
            (1, 'Smith', 'John', 'New York', 10.0),
            (2, 'Johnson', 'Mary', 'Los Angeles', 15.0),
            (3, 'Williams', 'David', 'Chicago', 5.0),
            (4, 'Brown', 'Sarah', 'New York', 20.0),
            (5, 'Jones', 'Chris', 'Miami', 8.0)
        ]
        cursor.executemany('INSERT INTO client VALUES (?, ?, ?, ?, ?)', clients)
        print("Client data Inserted")

        agents = [
            (101, 'Wilson', 'Robert', 'North', None, 15.0),  # 主管
            (102, 'Taylor', 'Jennifer', 'North', 101, 12.0),  # 下属
            (103, 'Anderson', 'Michael', 'South', None, 18.0),  # 主管
            (104, 'Thomas', 'Lisa', 'South', 103, 10.0),  # 下属
            (105, 'Martin', 'James', 'East', None, 16.0)  # 主管
        ]
        cursor.executemany('INSERT INTO agent VALUES (?, ?, ?, ?, ?, ?)', agents)
        print("Agent data Inserted")

        orders = [
            (1001, 1, 102, '2024-01-15', 'Laptop', 1200.00),
            (1002, 2, 104, '2024-01-16', 'Printer', 300.00),
            (1003, 3, 102, '2024-01-17', 'Monitor', 450.00),
            (1004, 1, 104, '2024-01-18', 'Keyboard', 80.00),
            (1005, 4, 105, '2024-01-19', 'Tablet', 600.00),
            (1006, 5, 102, '2024-01-20', 'Mouse', 25.00)
        ]
        cursor.executemany('INSERT INTO order_table VALUES (?, ?, ?, ?, ?, ?)', orders)
        print("order data inserted")

        conn.commit()
        print("ALL data inserted")

    except sqlite3.IntegrityError as e:
        print(f"error on constraints: {e}")
    except sqlite3.Error as e:
        print(f"error on data insert: {e}")
    finally:
        conn.close()


def test_constraints():
    conn = sqlite3.connect('Shop_Exercise.db')
    cursor = conn.cursor()

    print("\n" + "=" * 50)
    print("start test constraints")
    print("=" * 50)

    test_cases = [
        {
            'name': 'meaningless discount',
            'sql': "INSERT INTO client VALUES (99, 'Test', 'User', 'City', 150.0)",
            'expected': 'fail',
            'reason': 'not in 0-100'
        },
        {
            'name': 'non-existent foreign keys test',
            'sql': "INSERT INTO order_table VALUES (9999, 999, 999, '2024-01-01', 'Test', 100.0)",
            'expected': 'fail',
            'reason': 'Referencing non-existent clients and agents'
        },
        {
            'name': 'Negative Amount Test',
            'sql': "INSERT INTO order_table VALUES (9998, 1, 102, '2024-01-01', 'Test', -50.0)",
            'expected': 'fail',
            'reason': 'The order amount must be a positive number.'
        },
        {
            'name': 'Data validation test',
            'sql': "INSERT INTO client VALUES (100, 'Valid', 'User', 'Seattle', 25.5)",
            'expected': 'success',
            'reason': 'all condition conformed'
        }
    ]

    for test in test_cases:
        try:
            cursor.execute(test['sql'])
            if test['expected'] == '成功':
                print(f"✓ {test['name']}: success - {test['reason']}")
            else:
                print(f"✗ {test['name']}: fail - constrain failed")
                conn.rollback()
        except sqlite3.IntegrityError:
            if test['expected'] == 'fail':
                print(f"✓ {test['name']}: success - {test['reason']}")
            else:
                print(f"✗ {test['name']}: fail - unexpected constraint error")
            conn.rollback()
        except sqlite3.Error as e:
            print(f"✗ {test['name']}: error - {e}")
            conn.rollback()

    conn.close()


def display_data():
    conn = sqlite3.connect('Shop_Exercise.db')
    cursor = conn.cursor()

    print("\n" + "=" * 50)
    print("Database Content Preview")
    print("=" * 50)

    print("\nclient table (client):")
    cursor.execute("SELECT * FROM client")
    for row in cursor.fetchall():
        print(f"  ID: {row[0]}, name: {row[1]} {row[2]}, city: {row[3]}, discount: {row[4]}%")

    print("\nagent table (agent):")
    cursor.execute("SELECT * FROM agent")
    for row in cursor.fetchall():
        supervisor_info = f", supervisor: {row[4]}" if row[4] else ", supervisor: null"
        print(f"  ID: {row[0]}, name: {row[1]} {row[2]}, region: {row[3]}{supervisor_info}, salary: {row[5]}%")

    print("\norder table (order_table):")
    cursor.execute("SELECT * FROM order_table")
    for row in cursor.fetchall():
        print(
            f"  order: {row[0]}, customer ID: {row[1]}, agent ID: {row[2]}, date: {row[3]}, products: {row[4]}, amount: ${row[5]}")

    conn.close()


def cleanup():
    if os.path.exists('Shop_Exercise.db'):
        os.remove('Shop_Exercise.db')
        print("\nDataBase Cleaned")


# main
if __name__ == "__main__":
    try:
        # 1. initialize_database
        print("step1: Initialize database")
        initialize_database()

        # 2. insert sample data
        print("\nstep2: nsert sample data")
        insert_sample_data()

        # 3. display
        display_data()

        # 4. test constraints
        test_constraints()

        print("\n" + "=" * 50)
        print("Database Initialized！")
        print("=" * 50)

    except Exception as e:
        print(f"ERROR: {e}")

    finally:
        # cleanup()
        pass