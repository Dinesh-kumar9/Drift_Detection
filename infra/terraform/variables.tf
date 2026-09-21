variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment: local | staging | production"
  type        = string
  default     = "staging"

  validation {
    condition     = contains(["local", "staging", "production"], var.environment)
    error_message = "environment must be one of: local, staging, production"
  }
}

variable "project_name" {
  description = "Project identifier used in resource naming"
  type        = string
  default     = "mlops-observability"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "db_name" {
  description = "Postgres database name"
  type        = string
  default     = "mlops_db"
}

variable "db_username" {
  description = "Postgres master username"
  type        = string
  default     = "mlops"
  sensitive   = true
}

variable "backend_image_tag" {
  description = "Docker image tag deployed to ECS"
  type        = string
  default     = "latest"
}
