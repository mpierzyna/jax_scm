Fixtures for end-to-end tests are validated outputs from validation runs.
They should be copied instead of linked (like in earlier versions) so that
fixtures are unaffected by reruns of the validation cases. Change detection
may otherwise fail.

## Updating fixtures

Each fixture directory contains an `update.sh` script. To update a fixture,
run its script from the fixture directory:

```bash
cd tests/fixtures/gabls1 && bash update.sh
cd tests/fixtures/andren1994 && bash update.sh
cd tests/fixtures/wangara && bash update.sh
```

Each script cds into the corresponding `validation/` directory, runs `run.py`
there, and copies the output back. **Before updating a fixture, visually inspect
the HTML report produced in the validation directory** to confirm the run looks
physically correct. The fixture is the reference the e2e tests compare against,
so it should only be updated when the new output is intentionally different
(e.g. after a physics or numerics change that has been validated).
