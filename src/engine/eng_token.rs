use tiktoken_rs::{CoreBPE, o200k_base};

pub struct EnglishTokenizer {
    bpe: CoreBPE,
}

impl EnglishTokenizer {
    pub fn new() -> Self {
        Self {
            bpe: o200k_base().unwrap(),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.bpe.encode_with_special_tokens(text)
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        self.bpe.decode(tokens).unwrap()
    }
}

impl Default for EnglishTokenizer {
    fn default() -> Self {
        Self::new()
    }
}
