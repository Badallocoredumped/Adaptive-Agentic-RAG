export type Route = 'sql' | 'text' | 'hybrid';

export interface SubTask {
  sub_query: string;
  route: Route;
}

export interface SynthesisResult {
  answer: string;
  needs_clarification: boolean;
  reason?: string;
  question?: string;
  sources: string[];
  latency?: number;
}

export interface QueryRequest {
  user_query: string;
}

export interface IngestResponse {
  ingested_count: number;
}
