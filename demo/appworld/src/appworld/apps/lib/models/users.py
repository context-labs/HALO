from typing import Protocol, TypeVar

from appworld.apps.lib.models.orm import SQLModel


TSQLModel = TypeVar("TSQLModel", bound="SQLModel")


class UserLike(Protocol):
    email: str
    name: str
    phone_number: str


def user_email_name(user: UserLike) -> dict[str, str]:
    return {"email": user.email, "name": user.name}


def user_phone_number_name(user: UserLike) -> dict[str, str]:
    return {"phone_number": user.phone_number, "name": user.name}


def find_user_from_another(user_class: type[TSQLModel], user: SQLModel) -> TSQLModel | None:
    """Find user in one model class (app) from user object in another model class (app)."""
    if "email" in user_class.field_names():
        if not hasattr(user, "email"):
            raise ValueError(f"User {user} does not have email.")
        return user_class.find_one(email=user.email)
    elif "phone_number" in user_class.field_names():
        if not hasattr(user, "phone_number"):
            raise ValueError(f"User {user} does not have phone number.")
        return user_class.find_one(phone_number=user.phone_number)
    raise ValueError(f"User {user} does not have email or phone number.")


def find_user_from_another_or_raise(user_class: type[TSQLModel], user: SQLModel) -> TSQLModel:
    """Find user in one model class (app) from user object in another model class (app)."""
    found_user = find_user_from_another(user_class, user)
    if found_user is None:
        raise ValueError(f"User {user} not found in {user_class}.")
    return found_user
