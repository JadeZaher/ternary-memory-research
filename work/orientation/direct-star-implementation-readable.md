
:root {
      --tab-size-preference: 4;
    }
pre, code {
      tab-size: var(--tab-size-preference);
    }
GitHub - mitchuski/star-key · GitHub
Skip to content
Navigation Menu
Sign in
Appearance settings
Platform
AI CODE CREATION
GitHub Copilot
Write better code with AI
GitHub Copilot app
Direct agents from issue to merge
MCP Registry
Integrate external tools
DEVELOPER WORKFLOWS
Actions
Automate any workflow
Codespaces
Instant dev environments
Issues
Plan and track work
Code Review
Manage code changes
Code Quality
Enforce quality at merge
APPLICATION SECURITY
GitHub Advanced Security
Find and fix vulnerabilities
Code security
Secure your code as you build
Secret protection
Stop leaks before they start
EXPLORE
Why GitHub
Documentation
Blog
Changelog
Marketplace
View all features
Solutions
BY COMPANY SIZE
Enterprises
Small and medium teams
Startups
Nonprofits
BY USE CASE
App Modernization
DevSecOps
DevOps
CI/CD
View all use cases
BY INDUSTRY
Healthcare
Financial services
Manufacturing
Government
View all industries
View all solutions
Resources
EXPLORE BY TOPIC
AI
Software Development
DevOps
Security
View all topics
EXPLORE BY TYPE
Customer stories
Events & webinars
Ebooks & reports
Business insights
GitHub Skills
SUPPORT & SERVICES
Documentation
Customer support
Community forum
Trust center
Partners
View all resources
Open Source
COMMUNITY
GitHub Sponsors
Fund open source developers
PROGRAMS
Security Lab
Maintainer Community
GitHub Stars
Archive Program
REPOSITORIES
Topics
Trending
Collections
Enterprise
ENTERPRISE SOLUTIONS
Enterprise platform
AI-powered developer platform
AVAILABLE ADD-ONS
GitHub Advanced Security
Enterprise-grade security features
Copilot for Business
Enterprise-grade AI features
Premium Support
Enterprise-grade 24/7 support
Pricing
Search
/
Sign in
Sign up
Appearance settings
You signed in with another tab or window. 
Reload
 to refresh your session.
You signed out in another tab or window. 
Reload
 to refresh your session.
You switched accounts on another tab or window. 
Reload
 to refresh your session.
Dismiss alert
{{ message }}
mitchuski
/
star-key
Public
Notifications
You must be signed in to change notification settings
Fork
0
Star
0
Code
Issues
0
Pull requests
0
Actions
Projects
Security and quality
0
Insights
Additional navigation options
Code
Issues
Pull requests
Actions
Projects
Security and quality
Insights
main
Branches
Tags
Go to file
Code
Open more actions menu
Latest commit
History
302 Commits
302 Commits
Folders and files
Name
Name
Last commit message
Last commit date
.github/
workflows
.github/
workflows
docs
docs
packages
packages
pilot
pilot
scripts
scripts
.gitignore
.gitignore
CLAUDE.md
CLAUDE.md
README.md
README.md
STAR_PILOT.md
STAR_PILOT.md
package-lock.json
package-lock.json
package.json
package.json
tsconfig.base.json
tsconfig.base.json
tsconfig.json
tsconfig.json
View all files
Repository files navigation
README
More
 items
Star Key
Star is for trust.
Star Key is an agentprivacy fork of OpenVTC’s browser extension, developed through our collaboration in the Trust over IP community. We’re adapting it for Mages City: an agent community exploring Trust Spanning Protocol integration, DID-based messaging, shared knowledge and verifiable relationships.
Test today
The web reader lets you inspect a version-1 City Key, choose whether claimed vertices travel, and download an unverified compact reading. The illustration is editorial; compression is not a ZK proof. No sign-in, credential or City admission is created by this exercise.
The MV3 extension retains upstream VTA setup, consent and protocol controls. A complete verified VTA/RP encounter is still an integration gate. The provider pilot defaults to disabled and never falls back to simulated success.
Build
Use Node 24 or newer. Run 
npm ci
, 
npm test
, 
npm run build
 and 
node --test pilot/test.mjs
. Load 
packages/extension/dist
 unpacked in Chrome or Edge. Star has a distinct development extension ID: 
cdffmakpanghoamdnhbijolhncijajjc
.
For the portable website reader, run 
npm run build:star-preview
. 
packages/extension/dist-star-preview
 contains only the local reader and its artwork; it contains no extension provider bridge. Serve it under 
star-experiment/reading/
 and open 
star.html
. Preserve the experimental labels.
Ecosystem
Soulbis owns the boundary and Star perspective. Mages City hosts discovery and participation. Labs can reference Star Key as an experiment, with the web reader available before extension installation. Authentication, membership, permission and execution remain distinct decisions. A later explicit ceremony can attach a permitted receipt; no such receipt is issued today.
See 
pilot notes
, 
upstream documentation
, and 
pilot/pilot-config.json
. Real RP identity, independent session verification and enforced VTA persona selection remain required before showing a connected account.
Collaboration and upstream
Built from 
OpenVTC/vta-browser-plugin
 at 
7e1b3829b09057c13ca940b8d632d909f4569f7d
, with upstream history and attribution retained. Our work brings that foundation into the agentprivacy and Mages City experiments.
The core and TSP package metadata identify Apache-2.0. Upstream notices remain applicable; licensing coverage for the UI and agentprivacy artwork is still being clarified.
The public repository is 
mitchuski/star-key
. See 
the unified Star experience
 for the interface and integration direction. Native extension installation and live sign-in remain to be verified.
About
No description, website, or topics provided.
Resources
Readme
Activity
Stars
0
 stars
Watchers
0
 watching
Forks
0
forks
Report repository
Releases
Packages
Contributors
Languages
Footer
© 2026 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Community
Docs
Contact
Manage cookies
Do not share my personal information
You can’t perform that action at this time.

