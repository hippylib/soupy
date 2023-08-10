![Build Status](https://github.com/hippylib/soupy/actions/workflows/ci.yml/badge.svg)

# How to Contribute

SOUPy is managed by the hIPPYlib organization. 
The SOUPy team welcomes contributions at all levels: bugfixes, code
improvements, new capabilities, improved documentation, 
or new examples/tutorials.

Use a pull request (PR) toward the `soupy:main` branch to propose your
contribution. If you are planning significant code changes, or have any
questions, you should also open an [issue](https://github.com/hippylib/soupy/issues)
before issuing a PR. 

See the [Quick Summary](#quick-summary) section for the main highlights of our
GitHub workflow. For more details, consult the following sections and refer
back to them before issuing pull requests:

- [GitHub Workflow](#github-workflow)
  - [hIPPYlib Organization](#hippylib-organization)
  - [New Feature Development](#new-feature-development)
  - [Developer Guidelines](#developer-guidelines)
  - [Pull Requests](#pull-requests)
  - [Pull Request Checklist](#pull-request-checklist)
- [Automated Testing](#automated-testing)
- [Contact Information](#contact-information)

Contributing to SOUPy requires knowledge of Git.
If you are new to Git, see the [GitHub learning
resources](https://help.github.com/articles/git-and-github-learning-resources/).
SOUPy follows closely the framework of hIPPYlib, building on top of many of its features. 
To learn more about hIPPYlib, and inverse problems in general, we refer to the [hIPPYlib tutorial page](http://hippylib.github.io/tutorial).

*By submitting a pull request, you are affirming the* [Developer's Certificate of
Origin](#developers-certificate-of-origin-11) *at the end of this file.*


## Quick Summary

- We encourage you to [join the hIPPYlib organization](#hippylib-organization) and create
  development branches off `soupy:main`.
- Please follow the [developer guidelines](#developer-guidelines), in particular
  with regards to documentation and code styling.
- Pull requests  should be issued toward `soupy:main`. Make sure
  to check the items off the [Pull Request Checklist](#pull-request-checklist).
- After approval, SOUPy developers merge the PR in `soupy:main`.
- Don't hesitate to [contact us](#contact-information) if you have any questions.


## GitHub Workflow

The GitHub organization, https://github.com/hippylib, is the main developer hub for
the SOUPy project.

If you plan to make contributions or will like to stay up-to-date with changes
in the code, *we strongly encourage you to [join the hIPPYlib organization](#hippylib-organization)*.

This will simplify the workflow (by providing you additional permissions), and
will allow us to reach you directly with project announcements.


### hIPPYlib Organization

- Before you can start, you need a GitHub account, here are a few suggestions:
  + Create the account at: github.com/join.
  + For easy identification, please add your name and maybe a picture of you at: [https://github.com/settings/profile](https://github.com/settings/profile).
  + To receive notification, set a primary email at: [https://github.com/settings/emails](https://github.com/settings/emails).
  + For password-less pull/push over SSH, add your SSH keys at: [https://github.com/settings/keys](https://github.com/settings/keys).

- [Contact us](#contact-information) for an invitation to join the hIPPYlib GitHub
  organization.

- You should receive an invitation email, which you can directly accept.
  Alternatively, *after logging into GitHub*, you can accept the invitation at
  the top of [https://github.com/hippylib](https://github.com/hippylib).

- Consider making your membership public by going to [https://github.com/orgs/hippylib/people](https://github.com/orgs/hippylib/people)
  and clicking on the organization visibility dropbox next to your name.

- Project discussions and announcements will be posted at
  [https://github.com/orgs/hippylib/teams/everyone](https://github.com/orgs/hippylib/teams/everyone).

- The website for hIPPYlib is in the [web](https://github.com/hippylib/web) repository.

- The SOUPy source code is in the [soupy](https://github.com/hippylib/soupy)
  repository within the hIPPYlib organization.




### New Feature Development

- A new feature should be important enough that at least one person, the
  proposer, is willing to work on it and be its champion.

- The proposer creates a branch for the new feature (with suffix `-dev`), off
  the `main` branch, or another existing feature branch, for example:

  ```
  # Clone assuming you have setup your ssh keys on GitHub:
  git clone git@github.com:hippylib/soupy.git

  # Alternatively, clone using the "https" protocol:
  git clone https://github.com/hippylib/soupy.git

  # Create a new feature branch starting from "main":
  git checkout main
  git pull
  git checkout -b feature-dev

  # Work on "feature-dev", add local commits
  # ...

  # (One time only) push the branch to github and setup your local
  # branch to track the github branch (for "git pull"):
  git push -u origin feature-dev

  ```

- **We prefer that you create the new feature branch as a fork.**
  To allow SOUPy developers to edit the PR, please [enable upstream edits](https://help.github.com/articles/allowing-changes-to-a-pull-request-branch-created-from-a-fork/).

- The typical feature branch name is `new-feature-dev`, e.g. `taylor-dev`. While
  not frequent in SOUPy, other suffixes are possible, e.g. `-fix`, `-doc`, etc.

- For generic features relating to the hIPPYlib framework, consider developing the feature directly 
  in the [hIPPYlib repository](https://github.com/hippylib/hippylib).

### Developer Guidelines

- *Keep the code lean and as simple as possible*
  - Well-designed simple code is frequently more general and powerful.
  - Lean code base is easier to understand by new collaborators.
  - New features should be added only if they are necessary or generally useful.
  - Code must be compatible with Python 3.
  - When adding new features, consider adding an example in the `application` folder and/or a
    new notebook in the `tutorial` folder.
  - The preferred way to export solutions for visualization in paraview is using `dl.XDMFFile`

- *Keep the code general and reasonably efficient*
  - Main goal is fast prototyping for research.
  - When in doubt, generality wins over efficiency.
  - Respect the needs of different users (current and/or future).

- *Keep things separate and logically organized*
  - General usage features go in SOUPy (implemented in as much generality as
    possible), non-general features go into external apps/projects.
  - Inside SOUPy, compartmentalize between modeling, algorithms, utils, etc.
  - Contributions that are project-specific or have external dependencies are
    allowed (if they are of broader interest), but should be `#ifdef`-ed and not
    change the code by default.

- Code specifics
  - All significant new classes, methods and functions have sphinx-style
    documentation in source comments.
  - Code styling should resemble existing code.
  - When manually resolving conflicts during a merge, make sure to mention the
    conflicted files in the commit message.

### Pull Requests

- When your branch is ready for other developers to review / comment on
  the code, create a pull request towards `soupy:main`.

- Pull request typically have titles like:

     `Description [new-feature-dev]`

  for example:

     `Support for Taylor approximations [taylor-dev]`

  Note the branch name suffix (in square brackets).

- Titles may contain a prefix in square brackets to emphasize the type of PR.
  Common choices are: `[DON'T MERGE]`, `[WIP]` and `[DISCUSS]`, for example:

     `[DISCUSS] Support for Taylor approximations [taylor-dev]`

- Add a description, appropriate labels and assign yourself to the PR. The hIPPYlib
  team will add reviewers as appropriate.

- List outstanding TODO items in the description.

- Track the github workflow [continuous integration](#automated-testing)
  builds at the end of the PR. These should run clean, so address any errors as
  soon as possible.


### Pull Request Checklist

Before a PR can be merged, it should satisfy the following:

- [ ] CI runs without errors.
- [ ] Update `CHANGELOG`:
    - [ ] Is this a new feature users need to be aware of? New or updated application or tutorial?
    - [ ] Does it make sense to create a new section in the `CHANGELOG` to group with other related features?
- [ ] New examples/applications/tutorials:
    - [ ] All new examples/applications/tutorials run as expected.
    - [ ] If possible a *fast version* of the example/application/tutorial to the CI workflow
- [ ] New capability:
   - [ ] All significant new classes, methods and functions have sphinx-style documentation in source comments.
   - [ ] Add new examples/applications/tutorials to highlight the new capability.
   - [ ] For new classes, functions, or modules, edit the corresponding `.rst` file in the `doc` folder.
   - [ ] If this is a major new feature, consider mentioning in the short summary inside `README` *(rare)*.


## Automated Testing

We use github workflow to drive the default tests on the `main` and `feature`
branches. See the `.github/workflows/ci.yml` file.

Testing using github workflow should be kept lightweight.

- Tests on the `main` branch are triggered whenever a push is issued on this branch.
- Tests on the `feature` branch are triggered whenever the branch is submitted for PR. 


## Contact Information

- Contact the SOUPy team by posting to the [GitHub issue tracker](https://github.com/hippylib/soupy/issues).
  Please perform a search to make sure your question has not been answered already.
  
## Slack channel

The hIPPYlib organization's slack channel is a good resource to request and receive help with using hIPPYlib and SOUPy. Everyone is invited to read and take part in discussions. Discussions about development of new features in hIPPYlib also take place here. You can join our Slack community by filling in [this form](https://forms.gle/w8B7uKSXxdVCmfZ99). 

## [Developer's Certificate of Origin 1.1](https://developercertificate.org/)

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right
    to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my
    knowledge, is covered under an appropriate open source license and I have
    the right under that license to submit that work with modifications, whether
    created in whole or in part by me, under the same open source license
    (unless I am permitted to submit under a different license), as indicated in
    the file; or

(c) The contribution was provided directly to me by some other person who
    certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and
    that a record of the contribution (including all personal information I
    submit with it, including my sign-off) is maintained indefinitely and may be
    redistributed consistent with this project or the open source license(s)
    involved.
    
---    
> *Acknowledgement*: We thank the [MFEM team](https://github.com/mfem) for allowing us to use their
contributing guidelines file as template.