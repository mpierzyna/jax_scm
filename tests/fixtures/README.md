Fixtures for end-to-end tests are validated outputs from validation runs.
They should be copied instead of linked (like in earlier versions) so that
fixtures are unaffected by reruns of the validation cases. Change detection
may otherwise fail.