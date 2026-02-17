-- Crypto news digest cache
create table if not exists public.crypto_news_digest_cache (
  id uuid primary key default gen_random_uuid(),
  digest_date date not null,
  lang text not null check (lang in ('zh', 'en')),
  payload jsonb not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create unique index if not exists crypto_news_digest_cache_digest_date_lang_uq
  on public.crypto_news_digest_cache (digest_date, lang);

create index if not exists crypto_news_digest_cache_lang_digest_date_idx
  on public.crypto_news_digest_cache (lang, digest_date desc);

-- Optional: raw source events for observability
create table if not exists public.crypto_news_raw_events (
  id text primary key,
  source text not null,
  title text not null,
  url text not null,
  published_at timestamptz,
  normalized jsonb,
  ingested_at timestamptz not null default now()
);

-- Optional trigger to auto update updated_at
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists trg_crypto_news_digest_cache_updated_at on public.crypto_news_digest_cache;
create trigger trg_crypto_news_digest_cache_updated_at
before update on public.crypto_news_digest_cache
for each row execute function public.set_updated_at();
