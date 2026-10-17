use tree_sitter::Node;

use crate::parse::NodeTextExt;

pub(crate) fn declaration_identity(node: Node<'_>, source: &[u8]) -> Option<String> {
    if node.kind() != "identifier" {
        return None;
    }
    let signature = node.parent()?;
    if !is_constructor_signature(signature.kind()) {
        return None;
    }
    let identifiers = direct_identifiers(signature);
    if !identifiers.iter().take(2).any(|id| id.id() == node.id()) {
        return None;
    }
    let owner = identifiers[0].node_text(source);
    let name = identifiers.get(1).map_or("new", |n| n.node_text(source));
    (!owner.is_empty() && !name.is_empty()).then(|| format!("{owner}.{name}"))
}

pub(crate) fn is_constructor_declaration_identifier(node: Node<'_>) -> bool {
    if node.kind() != "identifier" {
        return false;
    }
    let Some(signature) = node.parent() else {
        return false;
    };
    if !is_constructor_signature(signature.kind()) {
        return false;
    }
    direct_identifiers(signature)
        .iter()
        .take(2)
        .any(|identifier| identifier.id() == node.id())
}

pub(crate) fn call_identity<'tree>(
    node: Node<'tree>,
    source: &[u8],
) -> Option<(Node<'tree>, Option<Node<'tree>>, String)> {
    match node.kind() {
        "constructor_invocation" | "new_expression" | "const_object_expression" => {
            let mut owner = None;
            let mut suffix = None;
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                match child.kind() {
                    "type_identifier" | "identifier" if owner.is_none() => owner = Some(child),
                    "type_identifier" | "identifier" if suffix.is_none() => suffix = Some(child),
                    _ => {}
                }
            }
            let owner = owner?;
            let owner_name = owner.node_text(source);
            let name = suffix.map_or("new", |n| n.node_text(source));
            (!owner_name.is_empty() && !name.is_empty())
                .then(|| (owner, suffix, format!("{owner_name}.{name}")))
        }
        "method_invocation" => {
            let function = node.child_by_field_name("function")?;
            let identifiers = if function.kind() == "identifier" {
                vec![function]
            } else {
                let mut identifiers = Vec::new();
                let mut cursor = function.walk();
                for child in function.named_children(&mut cursor) {
                    if child.kind() == "identifier" {
                        identifiers.push(child);
                    }
                }
                identifiers
            };
            let owner = *identifiers.first()?;
            let owner_name = owner.node_text(source);
            if !owner_name.chars().next()?.is_uppercase() {
                return None;
            }
            let suffix = identifiers.get(1).copied();
            let name = suffix.map_or("new", |n| n.node_text(source));
            (!owner_name.is_empty() && !name.is_empty())
                .then(|| (owner, suffix, format!("{owner_name}.{name}")))
        }
        _ => None,
    }
}

pub(crate) fn call_identity_for_capture(node: Node<'_>, source: &[u8]) -> Option<String> {
    let mut current = node;
    while let Some(parent) = current.parent() {
        if is_invocation(parent.kind()) {
            return call_identity(parent, source).map(|(_, _, name)| name);
        }
        if let Some((_, _, name)) = call_identity(parent, source) {
            return Some(name);
        }
        if matches!(
            parent.kind(),
            "function_body" | "block" | "source_file" | "program"
        ) {
            break;
        }
        current = parent;
    }
    None
}

fn is_invocation(kind: &str) -> bool {
    matches!(
        kind,
        "method_invocation"
            | "function_expression_invocation"
            | "constructor_invocation"
            | "new_expression"
            | "const_object_expression"
    )
}

fn direct_identifiers(node: Node<'_>) -> Vec<Node<'_>> {
    let mut cursor = node.walk();
    node.named_children(&mut cursor)
        .filter(|child| child.kind() == "identifier")
        .collect()
}

fn is_constructor_signature(kind: &str) -> bool {
    matches!(
        kind,
        "constructor_signature"
            | "constant_constructor_signature"
            | "factory_constructor_signature"
            | "redirecting_factory_constructor_signature"
    )
}
