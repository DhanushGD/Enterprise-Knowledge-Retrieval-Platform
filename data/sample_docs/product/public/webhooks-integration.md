# Setting up webhooks

Webhooks let OpenDesk notify your system when a ticket changes.

## Configure
1. Go to **Settings > Integrations > Webhooks**.
2. Add the HTTPS endpoint that should receive events.
3. Choose the events: ticket created, ticket updated, ticket closed.
4. Copy the signing secret and store it safely.

## Verifying signatures
Each request carries an `X-OpenDesk-Signature` header, an HMAC-SHA256 of the body
using your signing secret. Reject requests whose signature does not match.

## Retries
Failed deliveries are retried up to 5 times with exponential backoff.
After 5 failures the webhook is disabled and the admin is emailed.
