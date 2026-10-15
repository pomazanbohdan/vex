pub struct NavigationMenu {
    pub labels: Vec<String>,
}

pub fn retrieve_navigation_menu_metadata_from_server() -> NavigationMenu {
    NavigationMenu { labels: vec!["Home".into(), "Settings".into()] }
}

pub fn cache_navigation_menu_for_offline_use(menu: &NavigationMenu) -> usize {
    menu.labels.len()
}

pub fn validate_access_token(token: &str) -> bool {
    !token.is_empty()
}
