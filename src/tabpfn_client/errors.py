class TabPFNClientError(RuntimeError):
  pass


class ConfigError(TabPFNClientError):
  pass


class ServerError(TabPFNClientError):
  pass


class AuthError(TabPFNClientError):
  pass


class SerializationError(TabPFNClientError):
  pass


class ValidationError(TabPFNClientError):
  pass
