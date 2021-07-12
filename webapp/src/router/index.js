import Vue from 'vue'
import Router from 'vue-router'
import Home from '../views/Home.vue';
import Metric from '../views/Metric.vue';
import Filter from '../views/Filter.vue';
import Sample from '../views/Sample.vue';


Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'Home',
      component: Home,
    },
    {
      path: '/metric',
      name: 'Metric',
      component: Metric,
    },
    {
      path: '/filter',
      name: 'Filter',
      component: Filter,
    },
    {
      path: '/sample',
      name: 'Sample',
      component: Sample,
    },
    {
      path: '/about',
      name: 'About',
      // route level code-splitting
      // this generates a separate chunk (about.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () =>
        import(/* webpackChunkName: "about" */ '../views/About.vue'),
    }
  ]
})
