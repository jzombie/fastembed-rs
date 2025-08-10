// TODO: Remove; for prototyping

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use ort::execution_providers::CoreMLExecutionProvider;

fn main() -> anyhow::Result<()> {
    // 1. Create a list of providers using the documented API.
    let providers = vec![
        CoreMLExecutionProvider::default()
            // Provide a boolean argument to enable subgraphs
            .with_subgraphs(true)
            .build(),
        // The CPU provider is used automatically as a fallback
    ];

    // 2. Create InitOptions and pass the configured providers.
    let init_options =
        InitOptions::new(EmbeddingModel::BGESmallENV15).with_execution_providers(providers);

    // 3. Initialize the model.
    let mut model = TextEmbedding::try_new(init_options)?;

    println!("Model initialized successfully with CoreML provider!");

    let documents = vec!["This is a test document running on Apple Silicon."];
    let embeddings = model.embed(documents, None)?;

    println!("Successfully generated embeddings using the Neural Engine.");
    Ok(())
}
