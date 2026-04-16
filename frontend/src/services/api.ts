import axios from 'axios';
import type { SynthesisResult, IngestResponse } from '../types';

const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? 'http://localhost:8000',
});

export async function runQuery(user_query: string): Promise<SynthesisResult> {
  const { data } = await client.post<SynthesisResult>('/query', { user_query });
  return data;
}

export async function ingestDocuments(files: File[]): Promise<IngestResponse> {
  const form = new FormData();
  files.forEach((f) => form.append('files', f));
  const { data } = await client.post<IngestResponse>('/ingest', form);
  return data;
}
