def retrieve_navigation_menu_metadata(client):
    return client.fetch_navigation_menu()


def render_navigation_report(items):
    return [item["label"] for item in items]


def validate_access_token(token):
    return bool(token)
