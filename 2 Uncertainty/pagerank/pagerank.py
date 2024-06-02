import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    num_pages = len(corpus)
    distribution = {}

    # If the current page has no outgoing links, choose randomly among all pages
    if not corpus[page]:
        probability = 1 / num_pages
        return {page: probability for page in corpus}

    # Calculate probability for links from the current page
    link_probability = damping_factor / len(corpus[page])
    for link in corpus[page]:
        distribution[link] = link_probability

    # Calculate probability for all pages (including current) with 1 - damping_factor
    all_pages_probability = (1 - damping_factor) / num_pages
    for p in corpus:
        distribution.setdefault(p, 0)
        distribution[p] += all_pages_probability

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_ranks = {page: 0 for page in corpus}
    current_page = random.choice(list(corpus.keys()))

    for _ in range(n):
        distribution = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(list(distribution.keys()), weights=distribution.values())[0]
        page_ranks[current_page] += 1

    # Normalize page ranks to get proportions
    total_samples = sum(page_ranks.values())
    page_ranks = {page: count / total_samples for page, count in page_ranks.items()}

    return page_ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    initial_rank = 1 / num_pages
    page_ranks = {page: initial_rank for page in corpus}
    convergence_threshold = 0.001

    while True:
        new_ranks = {}
        for page in corpus:
            new_rank = (1 - damping_factor) / num_pages

            for linking_page, links in corpus.items():
                if not links:
                    # If a page has no links, distribute its rank evenly to all pages
                    new_rank += damping_factor * (page_ranks[linking_page] / num_pages)
                elif page in links:
                    new_rank += damping_factor * (page_ranks[linking_page] / len(links))

            new_ranks[page] = new_rank

        # Check for convergence
        if all(abs(new_ranks[page] - page_ranks[page]) < convergence_threshold for page in corpus):
            break

        page_ranks = new_ranks

    return page_ranks



if __name__ == "__main__":
    main()
