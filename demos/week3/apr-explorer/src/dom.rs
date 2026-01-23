//! Mock DOM for WASM Testing
//!
//! This module provides DOM abstractions that enable 100% test coverage
//! without requiring actual browser/web-sys dependencies.
//!
//! Probar: Visual feedback - Observable DOM state for testing

use std::collections::HashMap;

/// Represents a DOM element for testing
#[derive(Debug, Clone, PartialEq)]
pub struct DomElement {
    /// Element ID
    pub id: String,
    /// Element tag name
    pub tag: String,
    /// Text content
    pub text_content: String,
    /// Element attributes
    pub attributes: HashMap<String, String>,
    /// CSS classes
    pub classes: Vec<String>,
    /// Whether element is visible
    pub visible: bool,
    /// Child elements
    pub children: Vec<DomElement>,
}

impl Default for DomElement {
    fn default() -> Self {
        Self::new("div")
    }
}

impl DomElement {
    /// Creates a new DOM element with the given tag
    #[must_use]
    pub fn new(tag: &str) -> Self {
        Self {
            id: String::new(),
            tag: tag.to_string(),
            text_content: String::new(),
            attributes: HashMap::new(),
            classes: Vec::new(),
            visible: true,
            children: Vec::new(),
        }
    }

    /// Creates an element with an ID
    #[must_use]
    pub fn with_id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    /// Sets the text content
    #[must_use]
    pub fn with_text(mut self, text: &str) -> Self {
        self.text_content = text.to_string();
        self
    }

    /// Adds a class
    #[must_use]
    pub fn with_class(mut self, class: &str) -> Self {
        self.classes.push(class.to_string());
        self
    }

    /// Sets an attribute
    #[must_use]
    pub fn with_attr(mut self, key: &str, value: &str) -> Self {
        self.attributes.insert(key.to_string(), value.to_string());
        self
    }

    /// Adds a child element
    #[must_use]
    pub fn with_child(mut self, child: DomElement) -> Self {
        self.children.push(child);
        self
    }

    /// Sets visibility
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Sets text content
    pub fn set_text(&mut self, text: &str) {
        self.text_content = text.to_string();
    }

    /// Adds a class
    pub fn add_class(&mut self, class: &str) {
        if !self.classes.contains(&class.to_string()) {
            self.classes.push(class.to_string());
        }
    }

    /// Removes a class
    pub fn remove_class(&mut self, class: &str) {
        self.classes.retain(|c| c != class);
    }

    /// Checks if element has a class
    #[must_use]
    pub fn has_class(&self, class: &str) -> bool {
        self.classes.contains(&class.to_string())
    }

    /// Gets an attribute value
    #[must_use]
    pub fn get_attr(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).map(|s| s.as_str())
    }
}

/// DOM events that can be dispatched
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DomEvent {
    /// Click event on an element
    Click { element_id: String },
    /// File drop event
    FileDrop {
        element_id: String,
        file_name: String,
        file_size: usize,
    },
    /// Input event with new value
    Input { element_id: String, value: String },
    /// Key press event
    KeyPress {
        key: String,
        ctrl: bool,
        shift: bool,
    },
}

impl DomEvent {
    /// Creates a click event
    #[must_use]
    pub fn click(element_id: &str) -> Self {
        Self::Click {
            element_id: element_id.to_string(),
        }
    }

    /// Creates a file drop event
    #[must_use]
    pub fn file_drop(element_id: &str, file_name: &str, file_size: usize) -> Self {
        Self::FileDrop {
            element_id: element_id.to_string(),
            file_name: file_name.to_string(),
            file_size,
        }
    }

    /// Creates an input event
    #[must_use]
    pub fn input(element_id: &str, value: &str) -> Self {
        Self::Input {
            element_id: element_id.to_string(),
            value: value.to_string(),
        }
    }

    /// Creates a key press event
    #[must_use]
    pub fn key_press(key: &str) -> Self {
        Self::KeyPress {
            key: key.to_string(),
            ctrl: false,
            shift: false,
        }
    }
}

/// Mock DOM for testing APR explorer without browser
#[derive(Debug)]
pub struct MockDom {
    /// Root element
    pub root: DomElement,
    /// Elements by ID for quick lookup
    elements: HashMap<String, DomElement>,
    /// Event history for verification
    event_history: Vec<DomEvent>,
}

impl Default for MockDom {
    fn default() -> Self {
        Self::new()
    }
}

impl MockDom {
    /// Creates a new mock DOM
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: DomElement::new("div").with_id("root"),
            elements: HashMap::new(),
            event_history: Vec::new(),
        }
    }

    /// Creates an APR explorer DOM structure
    #[must_use]
    pub fn apr_explorer() -> Self {
        let mut dom = Self::new();

        // Status display
        let status = DomElement::new("div")
            .with_id("status")
            .with_class("status")
            .with_text("Ready");

        // Drop zone
        let drop_zone = DomElement::new("div")
            .with_id("drop-zone")
            .with_class("drop-zone");

        // Metrics panel
        let load_time = DomElement::new("span").with_id("load-time").with_text("-");
        let file_size = DomElement::new("span").with_id("file-size").with_text("-");
        let tensor_count = DomElement::new("span")
            .with_id("tensor-count")
            .with_text("-");
        let alignment = DomElement::new("span").with_id("alignment").with_text("-");

        // Header info panel
        let header_info = DomElement::new("div")
            .with_id("header-info")
            .with_class("panel");

        // Metadata panel
        let metadata_info = DomElement::new("pre")
            .with_id("metadata-info")
            .with_class("panel");

        // Tensor list panel
        let tensor_list = DomElement::new("div")
            .with_id("tensor-list")
            .with_class("panel");

        // Build root
        dom.root = DomElement::new("div")
            .with_id("apr-explorer")
            .with_class("explorer-app")
            .with_child(status.clone())
            .with_child(drop_zone.clone())
            .with_child(load_time.clone())
            .with_child(file_size.clone())
            .with_child(tensor_count.clone())
            .with_child(alignment.clone())
            .with_child(header_info.clone())
            .with_child(metadata_info.clone())
            .with_child(tensor_list.clone());

        // Register elements
        dom.register_element(status);
        dom.register_element(drop_zone);
        dom.register_element(load_time);
        dom.register_element(file_size);
        dom.register_element(tensor_count);
        dom.register_element(alignment);
        dom.register_element(header_info);
        dom.register_element(metadata_info);
        dom.register_element(tensor_list);

        dom
    }

    /// Registers an element for ID lookup
    pub fn register_element(&mut self, element: DomElement) {
        if !element.id.is_empty() {
            self.elements.insert(element.id.clone(), element);
        }
    }

    /// Gets an element by ID
    #[must_use]
    pub fn get_element(&self, id: &str) -> Option<&DomElement> {
        self.elements.get(id)
    }

    /// Gets a mutable element by ID
    pub fn get_element_mut(&mut self, id: &str) -> Option<&mut DomElement> {
        self.elements.get_mut(id)
    }

    /// Dispatches an event
    pub fn dispatch_event(&mut self, event: DomEvent) {
        self.event_history.push(event);
    }

    /// Gets the event history
    #[must_use]
    pub fn event_history(&self) -> &[DomEvent] {
        &self.event_history
    }

    /// Clears event history
    pub fn clear_event_history(&mut self) {
        self.event_history.clear();
    }

    /// Updates element text by ID
    pub fn set_element_text(&mut self, id: &str, text: &str) {
        if let Some(elem) = self.elements.get_mut(id) {
            elem.set_text(text);
        }
    }

    /// Gets element text by ID
    #[must_use]
    pub fn get_element_text(&self, id: &str) -> Option<&str> {
        self.elements.get(id).map(|e| e.text_content.as_str())
    }

    /// Adds a child element to a parent
    pub fn append_child(&mut self, parent_id: &str, child: DomElement) {
        let child_id = child.id.clone();
        if let Some(parent) = self.elements.get_mut(parent_id) {
            parent.children.push(child.clone());
        }
        if !child_id.is_empty() {
            self.elements.insert(child_id, child);
        }
    }

    /// Clears children of an element
    pub fn clear_children(&mut self, id: &str) {
        let child_ids: Vec<String> = self
            .elements
            .get(id)
            .map(|elem| {
                elem.children
                    .iter()
                    .filter(|c| !c.id.is_empty())
                    .map(|c| c.id.clone())
                    .collect()
            })
            .unwrap_or_default();

        for child_id in child_ids {
            self.elements.remove(&child_id);
        }

        if let Some(elem) = self.elements.get_mut(id) {
            elem.children.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== DomElement tests =====

    #[test]
    fn test_dom_element_new() {
        let elem = DomElement::new("span");
        assert_eq!(elem.tag, "span");
        assert!(elem.id.is_empty());
    }

    #[test]
    fn test_dom_element_default() {
        let elem = DomElement::default();
        assert_eq!(elem.tag, "div");
    }

    #[test]
    fn test_dom_element_builder_pattern() {
        let elem = DomElement::new("div")
            .with_id("test")
            .with_text("Hello")
            .with_class("active")
            .with_attr("data-value", "42");

        assert_eq!(elem.id, "test");
        assert_eq!(elem.text_content, "Hello");
        assert!(elem.has_class("active"));
        assert_eq!(elem.get_attr("data-value"), Some("42"));
    }

    #[test]
    fn test_dom_element_with_child() {
        let child = DomElement::new("span").with_text("child");
        let parent = DomElement::new("div").with_child(child);
        assert_eq!(parent.children.len(), 1);
    }

    #[test]
    fn test_dom_element_visibility() {
        let mut elem = DomElement::new("div");
        assert!(elem.visible);
        elem.set_visible(false);
        assert!(!elem.visible);
    }

    #[test]
    fn test_dom_element_class_operations() {
        let mut elem = DomElement::new("div");
        elem.add_class("foo");
        elem.add_class("bar");
        elem.add_class("foo"); // duplicate
        assert_eq!(elem.classes.len(), 2);

        elem.remove_class("foo");
        assert!(!elem.has_class("foo"));
        assert!(elem.has_class("bar"));
    }

    // ===== DomEvent tests =====

    #[test]
    fn test_dom_event_click() {
        let event = DomEvent::click("btn");
        assert!(matches!(event, DomEvent::Click { element_id } if element_id == "btn"));
    }

    #[test]
    fn test_dom_event_file_drop() {
        let event = DomEvent::file_drop("drop-zone", "model.apr", 1024);
        assert!(
            matches!(event, DomEvent::FileDrop { element_id, file_name, file_size }
                if element_id == "drop-zone" && file_name == "model.apr" && file_size == 1024)
        );
    }

    #[test]
    fn test_dom_event_input() {
        let event = DomEvent::input("field", "value");
        assert!(matches!(event, DomEvent::Input { element_id, value }
            if element_id == "field" && value == "value"));
    }

    #[test]
    fn test_dom_event_key_press() {
        let event = DomEvent::key_press("Enter");
        assert!(matches!(event, DomEvent::KeyPress { key, ctrl, shift }
            if key == "Enter" && !ctrl && !shift));
    }

    // ===== MockDom tests =====

    #[test]
    fn test_mock_dom_new() {
        let dom = MockDom::new();
        assert_eq!(dom.root.id, "root");
        assert!(dom.event_history.is_empty());
    }

    #[test]
    fn test_mock_dom_apr_explorer() {
        let dom = MockDom::apr_explorer();
        assert!(dom.get_element("status").is_some());
        assert!(dom.get_element("drop-zone").is_some());
        assert!(dom.get_element("load-time").is_some());
        assert!(dom.get_element("tensor-list").is_some());
    }

    #[test]
    fn test_mock_dom_register_element() {
        let mut dom = MockDom::new();
        let elem = DomElement::new("span").with_id("test");
        dom.register_element(elem);
        assert!(dom.get_element("test").is_some());
    }

    #[test]
    fn test_mock_dom_get_element_mut() {
        let mut dom = MockDom::apr_explorer();
        if let Some(elem) = dom.get_element_mut("status") {
            elem.set_text("Loading...");
        }
        assert_eq!(dom.get_element_text("status"), Some("Loading..."));
    }

    #[test]
    fn test_mock_dom_event_dispatch() {
        let mut dom = MockDom::apr_explorer();
        dom.dispatch_event(DomEvent::click("drop-zone"));
        dom.dispatch_event(DomEvent::file_drop("drop-zone", "test.apr", 100));
        assert_eq!(dom.event_history().len(), 2);
    }

    #[test]
    fn test_mock_dom_clear_event_history() {
        let mut dom = MockDom::apr_explorer();
        dom.dispatch_event(DomEvent::click("drop-zone"));
        dom.clear_event_history();
        assert!(dom.event_history().is_empty());
    }

    #[test]
    fn test_mock_dom_append_child() {
        let mut dom = MockDom::apr_explorer();
        let child = DomElement::new("div")
            .with_id("tensor-0")
            .with_text("weights.0");
        dom.append_child("tensor-list", child);
        assert!(dom.get_element("tensor-0").is_some());
    }

    #[test]
    fn test_mock_dom_clear_children() {
        let mut dom = MockDom::apr_explorer();
        let child1 = DomElement::new("div").with_id("t1");
        let child2 = DomElement::new("div").with_id("t2");
        dom.append_child("tensor-list", child1);
        dom.append_child("tensor-list", child2);

        dom.clear_children("tensor-list");

        assert!(dom.get_element("t1").is_none());
        assert!(dom.get_element("t2").is_none());
    }
}
