export interface SearchResult {
  title: string;
  link: string;
  snippet: string;
  pagemap: Record<string, any>;
  datePublished: string;
  source: string;
}

export interface SearchResponse {
  results: SearchResult[];
}

export interface WebpageContent {
  url: string;
  title: string;
  description: string;
  markdown_content: string;
  meta_tags: Record<string, string>;
  stats: {
    word_count: number;
    approximate_chars: number;
  };
  content_preview: {
    first_500_chars: string;
  };
}

export interface WebpageAnalysisResponse {
  [url: string]: WebpageContent | { error: string };
}
