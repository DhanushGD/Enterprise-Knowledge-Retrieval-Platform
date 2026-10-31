import base64
import json
import time

import jwt
import pytest

from serving.security import (ALGORITHM, DUMMY_HASH, AuthError, ConfigError, check_secret,
                              decode_token, hash_password, issue_token, verify_password)

SECRET = "x" * 48


def test_password_roundtrip_and_wrong_password():
    h = hash_password("correct horse")
    assert verify_password("correct horse", h) and not verify_password("Correct horse", h)
    assert verify_password("pässwörd ✓", hash_password("pässwörd ✓"))


def test_same_password_gets_different_salts_and_the_plaintext_is_not_stored():
    a, b = hash_password("same"), hash_password("same")
    assert a != b and "same" not in a and a.startswith("scrypt$")


@pytest.mark.parametrize("junk", ["", "plaintext", "scrypt$1$2$3", "md5$a$b$c$d$e",
                                  "scrypt$x$8$1$AAAA$BBBB", "scrypt$16384$8$1$!!!$???"])
def test_malformed_stored_hashes_fail_closed_without_raising(junk):
    assert verify_password("anything", junk) is False


def test_dummy_hash_is_a_valid_hash_nobody_can_match():
    assert DUMMY_HASH.startswith("scrypt$") and not verify_password("", DUMMY_HASH)


# -------------------------------------------------------------------- tokens
def test_token_roundtrip_carries_only_identity():
    token, expires_in = issue_token("alice", SECRET, minutes=30)
    assert decode_token(token, SECRET) == "alice" and expires_in == 1800
    claims = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
    assert set(claims) == {"sub", "iss", "iat", "exp"}        # no role: it is looked up live


def test_expired_token_is_rejected():
    token, _ = issue_token("alice", SECRET, minutes=1, now=time.time() - 3600)
    with pytest.raises(AuthError):
        decode_token(token, SECRET)


def test_wrong_secret_tampering_and_garbage_are_rejected():
    token, _ = issue_token("alice", SECRET)
    with pytest.raises(AuthError):
        decode_token(token, "y" * 48)
    head, body, sig = token.split(".")
    forged = base64.urlsafe_b64encode(json.dumps(
        {"sub": "carol", "iss": "ekrp", "iat": 1, "exp": 9999999999}).encode()).rstrip(b"=").decode()
    for bad in (f"{head}.{forged}.{sig}", "not.a.jwt", "", "abc"):
        with pytest.raises(AuthError):
            decode_token(bad, SECRET)


def test_alg_none_and_other_algorithms_are_rejected():
    claims = {"sub": "carol", "iss": "ekrp", "iat": int(time.time()), "exp": int(time.time()) + 600}
    unsigned = jwt.encode(claims, key=None, algorithm="none")
    with pytest.raises(AuthError):
        decode_token(unsigned, SECRET)
    hs512 = jwt.encode(claims, SECRET, algorithm="HS512")
    with pytest.raises(AuthError):
        decode_token(hs512, SECRET)


def test_missing_expiry_issuer_or_subject_is_rejected():
    now = int(time.time())
    for claims in ({"sub": "a", "iss": "ekrp", "iat": now},                       # no exp: immortal
                   {"sub": "a", "iat": now, "exp": now + 99},                      # no issuer
                   {"iss": "ekrp", "iat": now, "exp": now + 99},                   # no subject
                   {"sub": "a", "iss": "someone-else", "iat": now, "exp": now + 99},
                   {"sub": "", "iss": "ekrp", "iat": now, "exp": now + 99}):
        with pytest.raises(AuthError):
            decode_token(jwt.encode(claims, SECRET, algorithm=ALGORITHM), SECRET)


def test_secret_check_warns_in_dev_and_refuses_in_production():
    assert check_secret(SECRET, "dev") == []
    assert check_secret("change-me-in-real-use", "dev")
    assert check_secret("short", "dev")
    with pytest.raises(ConfigError):
        check_secret("change-me-in-real-use", "production")
    with pytest.raises(ConfigError):
        check_secret("short", "Production")
    assert check_secret(SECRET, "production") == []
