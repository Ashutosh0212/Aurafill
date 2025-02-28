"""User configuration and authentication.

This module contains predefined user credentials and authentication utilities.
"""

# Predefined users with their roles and credentials
USERS = {
    "sumitk": {
        "password": "sumitk2024",  # You should change this to a secure password
        "role": "user"
    },
    "": {
        "password": "",  # You should change this to a secure password
        "role": "user"
    }
}

def authenticate(username: str, password: str) -> tuple[bool, str]:
    """Authenticate a user and return their role.
    
    Args:
        username: The username to authenticate
        password: The password to verify
        
    Returns:
        Tuple of (is_authenticated, role)
    """
    if username in USERS and USERS[username]["password"] == password:
        return True, USERS[username]["role"]
    return False, "" 