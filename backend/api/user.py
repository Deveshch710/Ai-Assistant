from fastapi import APIRouter

# Create a router instance
router = APIRouter()

# Define a route for /user
@router.get("/")
def user_info():
    return {"message": "This is the User API!"}

# You can add more routes related to users, such as profile, settings, etc.
