### Model performance

#### Answer accuracy (%)

```mermaid
xychart-beta
    x-axis ["Base Llama", "Llama-3.1-TB", "CPT + SFT", "SFT + RAG"]
    y-axis "Accuracy (%)" 0 --> 100
    bar [77.0, 80.6, 79.3, 71.6]
```

#### Penalized accuracy (%)

```mermaid
xychart-beta
    x-axis ["Base Llama", "Llama-3.1-TB", "CPT + SFT", "SFT + RAG"]
    y-axis "Penalized accuracy (%)" 0 --> 100
    bar [65.8, 71.0, 69.0, 57.5]
```

**Model labels:**
- **Base Llama:** Base Llama-3.1-8B
- **Llama-3.1-TB:** Fine-tuned TB-specialized model
- **CPT + SFT:** Continued pretraining + supervised fine-tuning
- **SFT + RAG:** Supervised fine-tuning + retrieval-augmented generation
