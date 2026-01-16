import asyncio
from datetime import datetime
from textwrap import dedent

from pandas import DataFrame

from everyrow import create_client, create_session
from everyrow.ops import dedupe
from everyrow.session import Session


async def call_dedupe(session: Session):
    # Deduplicate academic papers where duplicates may be:
    # - Same paper listed with arXiv ID and DOI
    # - Preprint versions vs published versions
    # - Slight title variations between venues
    # Requires research to determine if papers are actually the same work

    papers = DataFrame(
        [
            {
                "title": "Attention Is All You Need",
                "authors": "Vaswani et al.",
                "venue": "NeurIPS 2017",
                "identifier": "10.5555/3295222.3295349",
            },
            {
                "title": "Attention Is All You Need",
                "authors": "Vaswani, Shazeer, Parmar et al.",
                "venue": "arXiv",
                "identifier": "1706.03762",
            },
            {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                "authors": "Devlin et al.",
                "venue": "NAACL 2019",
                "identifier": "10.18653/v1/N19-1423",
            },
            {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "authors": "Devlin, Chang, Lee, Toutanova",
                "venue": "arXiv",
                "identifier": "1810.04805",
            },
            {
                "title": "Language Models are Few-Shot Learners",
                "authors": "Brown et al.",
                "venue": "NeurIPS 2020",
                "identifier": "GPT-3",
            },
            {
                "title": "GPT-3: Language Models are Few-Shot Learners",
                "authors": "Brown, Mann, Ryder et al.",
                "venue": "arXiv",
                "identifier": "2005.14165",
            },
            {
                "title": "Training language models to follow instructions with human feedback",
                "authors": "Ouyang et al.",
                "venue": "NeurIPS 2022",
                "identifier": "InstructGPT",
            },
            {
                "title": "Constitutional AI: Harmlessness from AI Feedback",
                "authors": "Bai et al.",
                "venue": "arXiv",
                "identifier": "2212.08073",
            },
            {
                "title": "LLaMA: Open and Efficient Foundation Language Models",
                "authors": "Touvron et al.",
                "venue": "arXiv",
                "identifier": "2302.13971",
            },
            {
                "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
                "authors": "Touvron et al.",
                "venue": "arXiv",
                "identifier": "2307.09288",
            },
        ]
    )

    result = await dedupe(
        session=session,
        input=papers,
        equivalence_relation=dedent("""
            Two entries are duplicates if they represent the same research work, which requires
            verifying through research:

            - An arXiv preprint and its published conference/journal version are duplicates
            - Papers with slightly different titles but same core contribution are duplicates
            - Different author list formats (et al. vs full list) don't matter
            - Papers with different identifiers (arXiv ID vs DOI) may still be duplicates

            However, genuinely different papers (e.g., LLaMA 1 vs LLaMA 2) are NOT duplicates,
            even if authors and topics overlap. Research each paper to determine if they
            report the same findings or are distinct works.
        """),
    )
    print("Deduplicated Paper List:")
    print(result.data.to_string())
    print(f"\nArtifact ID: {result.artifact_id}")
    print(f"\nOriginal entries: {len(papers)}")
    print(f"Unique papers: {len(result.data)}")
    print(f"Duplicates removed: {len(papers) - len(result.data)}")


async def main():
    async with create_client() as client:
        session_name = f"Paper Deduplication {datetime.now().isoformat()}"
        async with create_session(client=client, name=session_name) as session:
            print(f"Session URL: {session.get_url()}")
            print("Deduplicating academic papers...")
            await call_dedupe(session)


if __name__ == "__main__":
    asyncio.run(main())
