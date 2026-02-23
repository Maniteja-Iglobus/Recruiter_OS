#!/usr/bin/env python
# ============================================================
# Admin Setup Script
# Run this after adding admin code to app.py
# Usage: python setup_admin.py
# ============================================================

import os
from datetime import datetime

def setup_admin():
    """Setup admin user"""
    print("=" * 60)
    print("🔐 Recruiter OS - Admin Setup Script")
    print("=" * 60)
    
    # Try to import from app
    try:
        from app import db, hash_password
        print("\n✅ Successfully imported from app.py")
    except Exception as e:
        print(f"\n❌ Error importing from app.py: {e}")
        print("\n⚠️ Make sure you have:")
        print("1. Added admin code to app.py")
        print("2. MongoDB is running")
        print("3. You're in the correct directory")
        return False
    
    # Check if admin already exists
    print("\n🔍 Checking for existing admin users...")
    existing_admin = db.admin_users.find_one({"username": "admin"})
    
    if existing_admin:
        print("⚠️ Admin user already exists!")
        print(f"  Created at: {existing_admin.get('created_at', 'Unknown')}")
        
        response = input("\nDo you want to:\n1. Use existing admin\n2. Create new admin with different username\nEnter 1 or 2: ")
        
        if response == "1":
            print("\n✅ Using existing admin user")
            print("Username: admin")
            return True
        elif response == "2":
            username = input("Enter new admin username: ").strip()
            if not username:
                print("❌ Username cannot be empty")
                return False
        else:
            print("❌ Invalid choice")
            return False
    else:
        username = "admin"
    
    # Get password
    print(f"\n📝 Creating admin user: '{username}'")
    password = input("Enter admin password (default: admin123): ").strip()
    if not password:
        password = "admin123"
    
    email = input("Enter admin email (default: admin@recruiterosapp.com): ").strip()
    if not email:
        email = "admin@recruiterosapp.com"
    
    # Create admin user
    try:
        admin_user = {
            "username": username,
            "password_hash": hash_password(password),
            "email": email,
            "role": "admin",
            "created_at": datetime.utcnow(),
            "is_active": True,
            "last_login": None
        }
        
        result = db.admin_users.insert_one(admin_user)
        
        print("\n" + "=" * 60)
        print("✅ ADMIN USER CREATED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\n📋 Admin Details:")
        print(f"   ID: {result.inserted_id}")
        print(f"   Username: {username}")
        print(f"   Password: {password}")
        print(f"   Email: {email}")
        print(f"   Created: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n🚀 Next Steps:")
        print(f"1. Start the backend: python -m uvicorn app:app --reload")
        print(f"2. Start the admin dashboard: streamlit run admin_dashboard_frontend.py")
        print(f"3. Login with:")
        print(f"   Username: {username}")
        print(f"   Password: {password}")
        
        print("\n⚠️  IMPORTANT:")
        print("   - Save these credentials somewhere safe")
        print("   - Change the password after first login in production")
        print("   - Never share admin credentials")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error creating admin user: {e}")
        return False

def main():
    """Main function"""
    try:
        success = setup_admin()
        
        if success:
            print("\n" + "=" * 60)
            print("✅ Setup completed successfully!")
            print("=" * 60)
            return 0
        else:
            print("\n" + "=" * 60)
            print("❌ Setup failed")
            print("=" * 60)
            return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
