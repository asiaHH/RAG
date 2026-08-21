import psycopg2

def get_scoped_connection(connection_string: str, user_id: str):
    """
    Opens a Postgres connection and sets the current user for RLS.  
    Use this wherever a request needs to be filtered/authorized by user, 
    including for INSERT statements (RLS also enforces the policy on writes).
    """
    conn = psycopg2.connect(connection_string)
    with conn.cursor() as cur:
        cur.execute("SET app.current_user_id = %s;", (user_id,))
    return conn