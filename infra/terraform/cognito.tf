# ─── AWS Cognito (Phase 4 — defined here as skeleton) ─────────────────────────

resource "aws_cognito_user_pool" "mlops" {
  name = "${var.project_name}-users-${var.environment}"

  password_policy {
    minimum_length    = 12
    require_lowercase = true
    require_uppercase = true
    require_numbers   = true
    require_symbols   = true
  }

  # Email-based MFA verification
  auto_verified_attributes = ["email"]

  # Custom attributes for RBAC role
  schema {
    name                     = "role"
    attribute_data_type      = "String"
    mutable                  = true
    developer_only_attribute = false
    string_attribute_constraints {
      min_length = 1
      max_length = 20
    }
  }

  tags = { Environment = var.environment, Project = var.project_name }
}

resource "aws_cognito_user_pool_client" "frontend" {
  name         = "${var.project_name}-frontend-${var.environment}"
  user_pool_id = aws_cognito_user_pool.mlops.id

  explicit_auth_flows = [
    "ALLOW_USER_PASSWORD_AUTH",
    "ALLOW_REFRESH_TOKEN_AUTH",
    "ALLOW_USER_SRP_AUTH",
  ]

  access_token_validity  = 1    # hours
  refresh_token_validity = 30   # days
  token_validity_units {
    access_token  = "hours"
    refresh_token = "days"
  }
}

output "cognito_user_pool_id" {
  value = aws_cognito_user_pool.mlops.id
}

output "cognito_client_id" {
  value = aws_cognito_user_pool_client.frontend.id
}
