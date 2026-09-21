# ─── RDS PostgreSQL ──────────────────────────────────────────────────────────

resource "aws_db_subnet_group" "mlops" {
  name       = "${var.project_name}-db-subnet-${var.environment}"
  subnet_ids = aws_subnet.private[*].id
  tags       = { Environment = var.environment }
}

resource "aws_security_group" "rds" {
  name   = "${var.project_name}-rds-sg-${var.environment}"
  vpc_id = aws_vpc.mlops.id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.mlops.cidr_block]  # VPC-internal only
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "mlops" {
  identifier        = "${var.project_name}-${var.environment}"
  engine            = "postgres"
  engine_version    = "16"
  instance_class    = var.db_instance_class
  allocated_storage = 20
  storage_encrypted = true

  db_name  = var.db_name
  username = var.db_username
  # password fetched from AWS Secrets Manager — not stored here
  manage_master_user_password = true

  db_subnet_group_name   = aws_db_subnet_group.mlops.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  multi_az               = var.environment == "production"
  skip_final_snapshot    = var.environment != "production"
  deletion_protection    = var.environment == "production"

  backup_retention_period = 7
  backup_window           = "03:00-04:00"

  tags = { Environment = var.environment, Project = var.project_name }
}
