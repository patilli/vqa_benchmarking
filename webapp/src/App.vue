<template>
  <div id="app">
    <b-navbar size="lg" toggleable="lg" type="dark" variant="dark">
      <b-navbar-brand to="/">VQA Benchmarking</b-navbar-brand>

      <b-collapse id="nav-collapse" is-nav>
        <b-navbar-toggle target="nav-collapse"></b-navbar-toggle>
        <b-navbar-nav>
            <b-nav-item href="#">Paper</b-nav-item>
        </b-navbar-nav>

        <!-- Right aligned nav items -->
        <b-navbar-nav class="ml-auto">
            <b-nav-item to="About">About</b-nav-item>
        </b-navbar-nav>
      </b-collapse>
    </b-navbar>
    <router-view/>
  </div>
</template>

<script>
export default {
  name: 'App'
  , methods: {
    navHome() {
      this.$router.push({
          name: "Home",
          props: true,
        })
      }
    }
}
</script>
