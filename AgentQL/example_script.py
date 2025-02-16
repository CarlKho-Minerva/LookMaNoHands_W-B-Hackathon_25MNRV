#!/usr/bin/env python3

import os
import sys
import time
from typing import Optional

import agentql
from playwright.sync_api import sync_playwright, Page as PlaywrightPage
from agentql.ext.playwright.sync_api import Page


def wait_for_search_results(page: Page, max_attempts: int = 10) -> bool:
    """Wait for search results to appear with retry logic."""
    for _ in range(max_attempts):
        response = page.query_elements(SEARCH_RESULTS_QUERY)
        if (
            hasattr(response.modal.search_results, "items")
            and len(response.modal.search_results.items) > 0
        ):
            return True
        time.sleep(0.5)
    return False


def main():
    try:
        if not os.getenv("AGENTQL_API_KEY"):
            print("Error: AgentQL API key is not set.")
            print(
                "Please set your API key using: export AGENTQL_API_KEY='your-api-key'"
            )
            print("Or sign up at https://dev.agentql.com to get an API key")
            sys.exit(1)

        with sync_playwright() as p:
            # Launch browser with correct options
            browser = p.chromium.launch(
                headless=False, args=["--ignore-certificate-errors"]
            )

            # Create context with SSL error handling
            context = browser.new_context(ignore_https_errors=True)

            try:
                # Create and wrap page
                page = agentql.wrap(context.new_page())
                page.goto("https://docs.agentql.com/quick-start")

                # Find and click search button
                search_button = page.get_by_prompt("search button")
                if search_button:
                    search_button.click()
                else:
                    print("Error: Could not find search button")
                    return

                # Wait for modal to appear
                page.wait_for_timeout(1000)

                # Get modal's search input and search
                response = page.query_elements(SEARCH_BOX_QUERY)
                if hasattr(response.modal, "search_box"):
                    response.modal.search_box.type("Quick Start")
                else:
                    print("Error: Could not find search box")
                    return

                # Wait for results with timeout
                if wait_for_search_results(page):
                    response = page.query_elements(SEARCH_RESULTS_QUERY)
                    if len(response.modal.search_results.items) > 0:
                        response.modal.search_results.items[0].click()
                    else:
                        print("No search results found")
                else:
                    print("Timeout waiting for search results")

                # Demo delay
                page.wait_for_timeout(5000)

            finally:
                browser.close()

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
