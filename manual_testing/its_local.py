"""Throwaway IndyKite Token Service (ITS) for testing the ik_token leg by hand.

Two steps, run from the repo root:

    uv run --locked python -m manual_testing.its_local setup /tmp/its
    docker run -d --name its-local -p 8103:8102 \\
        -v /tmp/its/token-service.yaml:/app/.configs/token-service.yaml:ro \\
        indykite/token-service:1.0.0 --config=/app/.configs/token-service.yaml
    uv run --locked python -m manual_testing.its_local mint /tmp/its --subject rebecca

`setup` writes a token-service config with a fresh signing key, a fresh client secret (also
saved as `client-secret` in the directory) and a fake IdP the service trusts (its private key
stays in the directory too, for `mint`). `mint` signs a subject and an actor token with that
IdP key, exchanges them at the running service (RFC 8693) and prints the resulting delegation
- the value to send as `X-IK-Token`. It expires after 5 minutes.

Agent side: ITS_INTROSPECT_URL=http://127.0.0.1:8103/oauth2/introspect, ITS_CLIENT_ID as
printed by `setup`, ITS_CLIENT_SECRET=$(cat <dir>/client-secret), INTROSPECT_URL at the stub
(`make start_mock_introspect_service`, answers sub=rebecca). Then, one call per case with
manual_testing/call_agent.sh:

    INTENT_AGENT_IK_TOKEN=$(... mint /tmp/its)                      -> 200, subject rebecca,
                                     log "agent 'orchestrator' acting for subject 'rebecca'"
    (no INTENT_AGENT_IK_TOKEN)                                       -> 401 X-IK-Token required
    INTENT_AGENT_IK_TOKEN=garbage                                    -> 401 ik_token is not active
    INTENT_AGENT_IK_TOKEN=$(... mint /tmp/its --subject millicent)   -> 401 actor delegation not authorized
    docker stop its-local, any token                                 -> 401 ik_token introspection request failed

Check the happy-path log shows the ik_token introspection going to port 8103, not the stub's
9001 - the stub accepts any token, so a leftover ITS_INTROSPECT_URL pointing at it would make
every case above look green. Full walkthrough: README, "Manual test: ik_token (ITS) validation".
"""

import argparse
import json
import secrets
import sys
import time
from pathlib import Path

import httpx2
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

CLIENT_ID = "intent-agent"
CREDENTIAL_FILENAME = "client-secret"
IDP_ISSUER = "https://local-idp.test"
IDP_AUDIENCE = "agent-gateway"
IDP_KID = "idp-1"
TOKEN_TYPE_ACCESS = "urn:ietf:params:oauth:token-type:access_token"  # noqa: S105 - an RFC 8693 type URN
GRANT_TYPE_EXCHANGE = "urn:ietf:params:oauth:grant-type:token-exchange"
_TOKEN_LIFETIME_SECONDS = 600


def _generate_jwk(kid: str) -> tuple[dict, dict]:
    """A fresh RSA key pair as (private JWK, public JWK), both tagged for RS256 signing."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = json.loads(RSAAlgorithm.to_jwk(key))
    public = json.loads(RSAAlgorithm.to_jwk(key.public_key()))
    for jwk_dict in (private, public):
        jwk_dict.update(alg="RS256", kid=kid, use="sig")
    return private, public


def setup(directory: Path) -> None:
    """Write token-service.yaml and the fake IdP's private key into `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    signing_key, _ = _generate_jwk("its-1")
    idp_private, idp_public = _generate_jwk(IDP_KID)
    (directory / "idp-private.jwk").write_text(json.dumps(idp_private))
    client_secret = secrets.token_urlsafe(24)
    (directory / CREDENTIAL_FILENAME).write_text(client_secret)
    config = f"""---
service:
  name: token-service
  port: 8102
  environment: local
  log_level: debug
  base_url: https://its-local.test

idp:
  audiences:
    - {IDP_AUDIENCE}
    - {CLIENT_ID}
  token_ttl: 5m
  actor_type: Agent
  client_auth:
    type: client_secret_basic
    client_id: {CLIENT_ID}
    client_secret: {client_secret}
  signing_keys:
    - '{json.dumps(signing_key, separators=(",", ":"))}'

token_introspection:
  configurations:
    local-idp:
      token_types:
        - subject_token
        - actor_token
      matcher:
        type: jwt
        issuer: {IDP_ISSUER}
        audience: {IDP_AUDIENCE}
      validation:
        offline:
          keys:
            - '{json.dumps(idp_public, separators=(",", ":"))}'
"""
    (directory / "token-service.yaml").write_text(config)
    print(f"wrote {directory / 'token-service.yaml'}, idp-private.jwk and {CREDENTIAL_FILENAME}")  # noqa: T201 - CLI output
    print(  # noqa: T201 - CLI output
        f"agent settings: ITS_CLIENT_ID={CLIENT_ID} ITS_CLIENT_SECRET=$(cat {directory / CREDENTIAL_FILENAME})",
    )


def mint(directory: Path, *, its_url: str, subject: str, actor: str) -> str:
    """Sign subject/actor tokens with the fake IdP key and exchange them for a delegation."""
    key = RSAAlgorithm.from_jwk((directory / "idp-private.jwk").read_text())
    client_secret = (directory / CREDENTIAL_FILENAME).read_text().strip()
    now = int(time.time())
    base = {"iss": IDP_ISSUER, "aud": IDP_AUDIENCE, "iat": now, "exp": now + _TOKEN_LIFETIME_SECONDS}

    def sign(sub: str) -> str:
        return jwt.encode({**base, "sub": sub}, key, algorithm="RS256", headers={"kid": IDP_KID})

    response = httpx2.post(
        f"{its_url.rstrip('/')}/oauth2/token",
        auth=(CLIENT_ID, client_secret),
        data={
            "grant_type": GRANT_TYPE_EXCHANGE,
            "requested_token_type": TOKEN_TYPE_ACCESS,
            "subject_token": sign(subject),
            "subject_token_type": TOKEN_TYPE_ACCESS,
            "actor_token": sign(actor),
            "actor_token_type": TOKEN_TYPE_ACCESS,
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: `setup <dir>` or `mint <dir> [--its-url] [--subject] [--actor]`."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    setup_parser = commands.add_parser("setup", help="write token-service.yaml plus the fake IdP key")
    setup_parser.add_argument("directory", type=Path)
    mint_parser = commands.add_parser("mint", help="exchange a subject/actor pair for a delegation (ik_token)")
    mint_parser.add_argument("directory", type=Path)
    mint_parser.add_argument("--its-url", default="http://127.0.0.1:8103")
    mint_parser.add_argument("--subject", default="rebecca", help="sub of the delegation - match the IdP's sub")
    mint_parser.add_argument("--actor", default="orchestrator", help="requesting agent (act.sub)")
    args = parser.parse_args(argv)

    if args.command == "setup":
        setup(args.directory)
        return 0
    delegation = mint(args.directory, its_url=args.its_url, subject=args.subject, actor=args.actor)
    print(delegation)  # noqa: T201 - the token is the output
    return 0


if __name__ == "__main__":
    sys.exit(main())
