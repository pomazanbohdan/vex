use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::Value;

fn wanted_event(event: &Value) -> bool {
    matches!(
        event.as_str(),
        Some(
            "server.connected"
                | "server.status"
                | "server.error"
                | "analysis.outline"
                | "search.results"
        )
    )
}

// Deserialize every scalar, including escapes and numbers, without storing payloads.
// IgnoredAny's fast skip does not enforce all of Value's scalar/depth checks.
struct ValidatedDiscard;

impl<'de> Deserialize<'de> for ValidatedDiscard {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_any(Self)
    }
}

impl<'de> Visitor<'de> for ValidatedDiscard {
    type Value = Self;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E: serde::de::Error>(self, _: bool) -> std::result::Result<Self, E> {
        Ok(self)
    }
    fn visit_i64<E: serde::de::Error>(self, _: i64) -> std::result::Result<Self, E> {
        Ok(self)
    }
    fn visit_u64<E: serde::de::Error>(self, _: u64) -> std::result::Result<Self, E> {
        Ok(self)
    }
    fn visit_f64<E: serde::de::Error>(self, _: f64) -> std::result::Result<Self, E> {
        Ok(self)
    }
    fn visit_str<E: serde::de::Error>(self, _: &str) -> std::result::Result<Self, E> {
        Ok(self)
    }
    fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<Self, E> {
        Ok(self)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Self, A::Error> {
        while seq.next_element::<Self>()?.is_some() {}
        Ok(self)
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<Self, A::Error> {
        while map.next_key::<DiscardKey>()?.is_some() {
            map.next_value::<Self>()?;
        }
        Ok(self)
    }
}

struct DiscardKey;

impl<'de> Deserialize<'de> for DiscardKey {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct KeyVisitor;
        impl Visitor<'_> for KeyVisitor {
            type Value = DiscardKey;
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a JSON object key")
            }
            fn visit_str<E: serde::de::Error>(self, _: &str) -> std::result::Result<DiscardKey, E> {
                Ok(DiscardKey)
            }
        }
        deserializer.deserialize_str(KeyVisitor)
    }
}

#[derive(serde::Deserialize)]
#[serde(field_identifier, rename_all = "lowercase")]
enum HeaderField {
    Id,
    Event,
    #[serde(other)]
    Other,
}

struct MessageHeader {
    has_id: bool,
    wanted_event: bool,
}

impl<'de> Deserialize<'de> for MessageHeader {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct HeaderVisitor;
        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = MessageHeader;
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a JSON protocol object")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<MessageHeader, A::Error> {
                let mut header = MessageHeader {
                    has_id: false,
                    wanted_event: false,
                };
                while let Some(field) = map.next_key::<HeaderField>()? {
                    match field {
                        HeaderField::Id => {
                            map.next_value::<ValidatedDiscard>()?;
                            header.has_id = true;
                        }
                        HeaderField::Event => {
                            header.wanted_event = wanted_event(&map.next_value::<Value>()?)
                        }
                        HeaderField::Other => {
                            map.next_value::<ValidatedDiscard>()?;
                        }
                    }
                }
                Ok(header)
            }
        }
        deserializer.deserialize_map(HeaderVisitor)
    }
}

pub(super) fn parse_message(line: &str) -> serde_json::Result<Option<Value>> {
    if let Ok(MessageHeader {
        has_id: false,
        wanted_event: false,
    }) = serde_json::from_str(line)
    {
        return Ok(None);
    }
    let value: Value = serde_json::from_str(line)?;
    Ok((value.get("id").is_some() || wanted_event(&value["event"])).then_some(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Instant;

    fn legacy_message(line: &str) -> serde_json::Result<Option<Value>> {
        let value: Value = serde_json::from_str(line)?;
        Ok((value.get("id").is_some() || wanted_event(&value["event"])).then_some(value))
    }

    #[test]
    fn header_filter_preserves_payload_field_order_duplicates_and_nonobjects() {
        let lines = [
            r#"{"event":"analysis.errors","params":{"errors":[{"message":"🦊 \u00e9","location":{"offset":7}}]}}"#,
            r#"{"params":{"id":"not-a-response","event":"server.error"},"event":"analysis.errors"}"#,
            r#"{"params":{"outline":{"children":[{"element":{"name":"target"}}]}},"event":"analysis.outline"}"#,
            r#"{"result":{"element":{"name":"target"}},"id":"1"}"#,
            r#"{"id":null,"result":{}}"#,
            r#"{"event":"server.error","event":"ignored","params":{}}"#,
            r#"{"event":"ignored","event":"server.error","params":{}}"#,
            r#"{"id":1,"id":"last","params":{}}"#,
            r#"{"event":[],"params":{"values":[null,true,false,1,-1,1.5,"text",{}]}}"#,
            r#"{"\u0065vent":"server.status","params":{"analysis":{"isAnalyzing":true}}}"#,
            r#"["server.error",{"id":"1"}]"#,
            r#""server.error""#,
            "true",
            "null",
            "42",
            "{}",
        ];
        for line in lines {
            assert_eq!(
                parse_message(line).unwrap(),
                legacy_message(line).unwrap(),
                "{line}"
            );
        }
    }

    #[test]
    fn header_filter_rejects_invalid_ignored_payloads_like_value_parser() {
        let nested = format!(
            r#"{{"event":"ignored","params":{}0{}}}"#,
            "[".repeat(140),
            "]".repeat(140)
        );
        for line in [
            r#"{"event":"ignored","params":[1,]}"#,
            r#"{"event":"ignored"} trailing"#,
            r#"{"event":"ignored"}{}"#,
            r#"{"event":"ignored","params":1e400}"#,
            r#"{"event":"ignored","params":"\uD800"}"#,
            r#"{"event":"ignored","params":{"\uD800":0}}"#,
            r#"{"event":"ignored","params":"\x00"}"#,
            r#"{"event":"ignored","params":01}"#,
            r#"{"event":"server.error","params":[}"#,
            &nested,
        ] {
            let old = legacy_message(line).unwrap_err();
            let new = parse_message(line).unwrap_err();
            assert_eq!(new.to_string(), old.to_string(), "{line}");
        }
    }

    #[test]
    #[ignore = "manual release parser benchmark; no timing assertion"]
    fn profile_header_filter() {
        type Parser = fn(&str) -> serde_json::Result<Option<Value>>;
        let diagnostic = json!({"severity":"ERROR","type":"COMPILE_TIME_ERROR","message":"diagnostic detail ".repeat(16),"location":{"file":"lib/main.dart","offset":42,"length":7,"startLine":2,"startColumn":4}});
        let noise = json!({"event":"analysis.errors","params":{"file":"lib/main.dart","errors":vec![diagnostic; 2000]}}).to_string();
        let keep = r#"{"params":{"id":"s1","results":[],"isLast":true},"event":"search.results"}"#;
        assert_eq!(
            parse_message(&noise).unwrap(),
            legacy_message(&noise).unwrap()
        );
        assert_eq!(parse_message(keep).unwrap(), legacy_message(keep).unwrap());
        for iteration in 0..7 {
            let parsers: [(&str, Parser); 2] = if iteration % 2 == 0 {
                [("old", legacy_message), ("new", parse_message)]
            } else {
                [("new", parse_message), ("old", legacy_message)]
            };
            for (name, parser) in parsers {
                for (kind, line) in [("ignored", noise.as_str()), ("retained", keep)] {
                    let start = Instant::now();
                    for _ in 0..100 {
                        std::hint::black_box(parser(std::hint::black_box(line)).unwrap());
                    }
                    println!("PARSER_BENCH iteration={iteration} parser={name} kind={kind} bytes={} repeats=100 elapsed_us={}", line.len(), start.elapsed().as_micros());
                }
            }
        }
    }
}
