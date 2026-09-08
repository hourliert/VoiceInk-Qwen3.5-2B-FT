const API_ROOT = "/api/v1";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(API_ROOT + path, {
    ...init,
    signal,
    headers: {
      ...(init.body ? {"Content-Type": "application/json"} : {}),
      ...init.headers,
    },
  });
  if (!response.ok) {
    let message = String(response.status) + " " + response.statusText;
    try {
      const body = await response.json() as {error?: string};
      message = body.error || message;
    } catch {
      // Keep the HTTP status when the body is not JSON.
    }
    throw new ApiError(response.status, message);
  }
  const type = response.headers.get("content-type") || "";
  return (type.includes("application/json")
    ? await response.json()
    : await response.text()) as T;
}

export const api = {
  get<T>(path: string, signal?: AbortSignal) {
    return request<T>(path, {}, signal);
  },
  post<T>(path: string, body: unknown, signal?: AbortSignal) {
    return request<T>(
      path,
      {method: "POST", body: JSON.stringify(body)},
      signal,
    );
  },
};

export function queryString(values: Record<string, string | number | undefined>) {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(values)) {
    if (value !== undefined && value !== "") params.set(key, String(value));
  }
  const encoded = params.toString();
  return encoded ? "?" + encoded : "";
}
